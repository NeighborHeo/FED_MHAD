from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict
import argparse

import torch
from torch.utils.data import DataLoader

import flwr as fl
import utils
import copy
from dataset import PascalVocPartition
from model import vit_tiny_patch16_224
from early_stopper import EarlyStopper
from feddf import FedDF
from flwr.common import (parameters_to_ndarrays, ndarrays_to_parameters, FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env_comet'))

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

class CustomServer:
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace, experiment: Optional[Experiment] = None):
        self.experiment = experiment
        self.args = args  # add this line to save args as an instance attribute
        self.save_path = f"checkpoints/{args.port}/global"
        self.early_stopper = EarlyStopper(patience=5, delta=1e-4, checkpoint_dir=self.save_path)
        self.strategy = self.create_strategy(model, args.toy)
        self.model = model
        
    def fit_config(self, server_round: int) -> Dict[str, int]:
        return {
            "server_round": server_round,
            "batch_size": 16,
            "local_epochs": 1 if server_round < 2 else 2,
        }

    def evaluate_config(self, server_round: int) -> Dict[str, int]:
        val_steps = 5 if server_round < 4 else 10
        return {
            "val_steps": val_steps, 
            "server_round": server_round
        }

    def get_evaluate_fn(self, model: torch.nn.Module, toy: bool):
        # trainset, _, _ = utils.load_data()
        pascal_voc_partition = PascalVocPartition(args=self.args)
        trainset, testset = pascal_voc_partition.load_partition(-1)
        n_train = len(testset)
        print(f"n_train: {n_train}")
        if toy:
            valset = torch.utils.data.Subset(testset, range(n_train - 10, n_train))
        else:
            valset = torch.utils.data.Subset(testset, range(0, n_train))

        valLoader = DataLoader(valset, batch_size=self.args.batch_size)

        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
            model.load_state_dict(state_dict, strict=True)

            device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
            result = utils.test(model, valLoader, device=device, args=self.args)
            accuracy = result["acc"]
            loss = result["loss"]
            
            is_best_accuracy = self.early_stopper.is_best_accuracy(accuracy)
            if is_best_accuracy:
                filename = f"model_round{server_round}_acc{accuracy:.2f}_loss{loss:.2f}.pth"
                self.early_stopper.save_checkpoint(model, server_round, loss, accuracy, filename)

            if self.early_stopper.counter >= self.early_stopper.patience:
                print(f"Early stopping : {self.early_stopper.counter} >= {self.early_stopper.patience}")
                # todo : stop server
                
            if self.experiment is not None:
                result = {f"test_" + k: v for k, v in result.items()}
                self.experiment.log_metrics(result, step=server_round)
                
            return float(loss), {"accuracy": float(accuracy)}

        return evaluate
    
    def load_public_loader(self):
        pascal_voc_partition = PascalVocPartition(args=self.args)
        trainset, testset = pascal_voc_partition.load_partition(-1)
        n_train = len(testset)
        if self.args.toy:
            publicset = torch.utils.data.Subset(testset, range(n_train - 10, n_train))
        else:
            publicset = torch.utils.data.Subset(testset, range(0, n_train))
        publicLoader = DataLoader(publicset, batch_size=self.args.batch_size)
        return publicLoader
    
    def load_parameter(self, model: torch.nn.Module, parameters: fl.common.NDArrays)-> torch.nn.Module:
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict, strict=True)
        return model
    
    def get_fedavg_model(self, results: List[Tuple[ClientProxy, FitRes]]) -> torch.nn.Module:
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        fedavg_model = copy.deepcopy(self.model)
        self.load_parameter(fedavg_model, aggregate(weights_results))
        return fedavg_model        

    def get_logits(self, model: torch.nn.Module, publicLoader: DataLoader) -> torch.Tensor:
        """Infer logits from the given model."""
        device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        model.to(device)
        model.eval()

        logits_list = []
        m = torch.nn.Sigmoid()
        with torch.no_grad():
            for inputs, _ in publicLoader:
                inputs = inputs.to(device)
                logits = model(inputs)
                logits_list.append(m(logits).detach())

        return torch.cat(logits_list, dim=0)

    def ensemble_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """Ensemble logits from multiple models."""
        stacked_logits = torch.stack(logits_list, dim=0)
        ensembled_logits = torch.mean(stacked_logits, dim=0)
        return ensembled_logits
    
    def fit_aggregation_fn(self, results: List[Tuple[ClientProxy, FitRes]]) -> Parameters:
        """Aggregate the results of the training rounds."""

        # Step 1: Get the logits from all the models

        publicLoader = self.load_public_loader()
        fedavg_model = self.get_fedavg_model(results)

        logits_list = []
        for _, fit_res in results:
            copied_model = copy.deepcopy(self.model)
            copied_model = self.load_parameter(copied_model, parameters_to_ndarrays(fit_res.parameters))
            logits = self.get_logits(copied_model, publicLoader)
            logits_list.append(logits)

        # Step 2: Ensemble logits
        ensembled_logits = self.ensemble_logits(logits_list)

        # Step 3: Distill logits
        distilled_model = self.distill_training(fedavg_model, ensembled_logits, publicLoader)
        distilled_parameters = [val.cpu().numpy() for _, val in distilled_model.state_dict().items()]

        return distilled_parameters

    def distill_training(self, model: torch.nn.Module, ensembled_logits: torch.Tensor, publicLoader: DataLoader) -> torch.nn.Module:
        """Perform distillation training."""
        device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        model.to(device)
        model.train()

        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
        last_layer_name = list(model.named_children())[-1][0]
        parameters = [
            {'params': [p for n, p in model.named_parameters() if last_layer_name not in n], 'lr': self.args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if last_layer_name in n], 'lr': self.args.learning_rate*100},
        ]
        optimizer = torch.optim.SGD(params= parameters, lr=self.args.learning_rate, momentum=self.args.momentum, weight_decay=self.args.weight_decay)

        m = torch.nn.Sigmoid()
        for epoch in range(self.args.local_epochs):
            running_loss = 0.0
            for i, (inputs, _) in enumerate(publicLoader):
                inputs = inputs.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                outputs = m(outputs)
                loss = criterion(outputs, ensembled_logits[i * self.args.batch_size:(i + 1) * self.args.batch_size].to(device))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Distillation Epoch {epoch + 1}/{self.args.local_epochs}, Loss: {running_loss / len(publicLoader)}")
        return model
            
    def create_strategy(self, model: torch.nn.Module, toy: bool):
        model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        print("create_strategy")
        return FedDF(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            evaluate_fn=self.get_evaluate_fn(model, toy),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            fit_aggregation_fn=self.fit_aggregation_fn,
            initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        )

    def start_server(self, port: int, num_rounds: int):
        
        fl.server.start_server(
            server_address=f"0.0.0.0:{port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy,
        )
        
        
def init_argurments() -> argparse.Namespace:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start Flower server with experiment key.")
    parser.add_argument("--experiment_key", type=str, required=True, help="Experiment key")
    parser.add_argument("--toy", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--use_cuda", action="store_true", default=False, help="Set to true to use cuda. Default: False")
    parser.add_argument("--port", type=int, default=8080, required=False, help="Port to use for the server. Default: 8080")
    parser.add_argument("--learning_rate", type=float, default=0.00002, required=False, help="Learning rate. Default: 0.1")
    parser.add_argument("--momentum", type=float, default=0.9, required=False, help="Momentum. Default: 0.9")
    parser.add_argument("--weight_decay", type=float, default=1e-5, required=False, help="Weight decay. Default: 1e-5")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size. Default: 32")
    parser.add_argument("--datapath", type=str, default="~/.data/", required=False, help="dataset path")
    parser.add_argument("--alpha", type=float, default=0.5, required=False, help="alpha")
    parser.add_argument("--seed", type=int, default=1, required=False, help="seed")
    parser.add_argument("--num_rounds", type=int, default=100, required=False, help="Number of rounds to run. Default: 100")
    parser.add_argument("--dataset", type=str, default="pascal_voc", required=False, help="Dataset to use. Default: pascal_voc")
    parser.add_argument("--num_classes", type=int, default=20, required=False, help="Number of classes. Default: 10")
    parser.add_argument("--N_parties", type=int, default=5, required=False, help="Number of clients to use. Default: 10")
    parser.add_argument("--task", type=str, default="multilabel", required=False, help="Task to run. Default: multilabel")
    parser.add_argument("--noisy", type=float, default=0.0, required=False, help="Percentage of noisy data. Default: 0.0")
    parser.add_argument("--local_epochs", type=int, default=2, required=False, help="Number of local epochs. Default: 1")
    
    args = parser.parse_args()
    print("Experiment key:", args.experiment_key, "port:", args.port)
    return args
        
def init_comet_experiment(args: argparse.Namespace):
    experiment = Experiment(
        api_key = os.getenv('COMET_API_TOKEN'),
        project_name = os.getenv('COMET_PROJECT_NAME'),
        workspace= os.getenv('COMET_WORKSPACE'),
    )
    experiment.log_parameters(args)
    experiment.set_name(f"global_({args.port})_lr_{args.learning_rate}_bs_{args.batch_size}_rd_{args.num_rounds}_ap_{args.alpha}_ns_{args.noisy}")
    return experiment

def main() -> None:
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    utils.set_seed(42)
    args = init_argurments()
    
    # Initialize Comet experiment
    experiment = init_comet_experiment(args)

    # Load model
    model = vit_tiny_patch16_224(pretrained=True, num_classes=args.num_classes)
    custom_server = CustomServer(model, args, experiment)
    custom_server.start_server(args.port, args.num_rounds)

    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()