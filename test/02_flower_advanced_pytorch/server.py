from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse

import torch
from torch.utils.data import DataLoader

import flwr as fl
import utils
from early_stopper import EarlyStopper

import warnings
warnings.filterwarnings("ignore")

import os

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

class CustomServer:
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace, experiment: Optional[Experiment] = None):
        self.experiment = experiment
        self.strategy = self.create_strategy(model, args.toy)
        self.args = args
        self.save_path = f"checkpoints/{args.port}_server_best_models"
        self.early_stopper = EarlyStopper(patience=5, delta=1e-4, checkpoint_dir=self.save_path)
        
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
        trainset, _, _ = utils.load_data()

        n_train = len(trainset)
        if toy:
            valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
        else:
            valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

        valLoader = DataLoader(valset, batch_size=16)

        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
            model.load_state_dict(state_dict, strict=True)

            device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
            loss, accuracy = utils.test(model, valLoader, device=device)
            
            is_best_accuracy = self.early_stopper.is_best_accuracy(accuracy)
            if is_best_accuracy:
                filename = f"model_round{server_round}_acc{accuracy:.2f}_loss{loss:.2f}.pth"
                self.early_stopper.save_checkpoint(model, server_round, loss, accuracy, filename)

            if self.early_stopper.counter >= self.early_stopper.patience:
                print(f"Early stopping : {self.early_stopper.counter} >= {self.early_stopper.patience}")
                # todo : stop server
                
            if self.experiment is not None:
                self.experiment.log_metric("test_loss", loss, step=server_round)
                self.experiment.log_metric("test_accuracy", accuracy, step=server_round)
                
            return loss, {"accuracy": accuracy}

        return evaluate
    
    def create_strategy(self, model: torch.nn.Module, toy: bool):
        model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

        return fl.server.strategy.FedAvg(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=self.get_evaluate_fn(model, toy),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
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
    parser.add_argument("--rounds", type=int, default=10, required=False, help="Number of rounds to run. Default: 10")
    parser.add_argument("--learning_rate", type=float, default=0.1, required=False, help="Learning rate. Default: 0.1")
    parser.add_argument("--momentum", type=float, default=0.9, required=False, help="Momentum. Default: 0.9")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay. Default: 1e-4")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size. Default: 32")
    args = parser.parse_args()
    print("Experiment key:", args.experiment_key, "port:", args.port)
    return args
        
def init_comet_experiment(args: argparse.Namespace):
    experiment = Experiment(
        api_key = "3JenmgUXXmWcKcoRk8Yra0XcD",
        project_name = "test1",
        workspace="neighborheo"
    )
    experiment.log_parameters(args)
    experiment.set_name(f"central_({args.port})_lr_{args.learning_rate}_bs_{args.batch_size}_rd_{args.rounds}")
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
    model = utils.load_efficientnet(classes=10)
    custom_server = CustomServer(model, args, experiment)
    custom_server.start_server(args.port, args.rounds)
    
    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()



