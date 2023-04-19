import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse

import torch
from torch.utils.data import DataLoader

import flwr as fl
import utils
from dataset import PascalVocPartition
from model import vit_tiny_patch16_224
from early_stopper import EarlyStopper
from fedmhad import FedMHAD
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env_comet'))

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

class CustomServerManager:
    def __init__(self, model: torch.nn.Module, args: argparse.Namespace, experiment: Optional[Experiment] = None):
        utils.print_func_and_line()
        self.experiment = experiment
        self.args = args  # add this line to save args as an instance attribute
        self.save_path = f"checkpoints/{args.port}/global"
        self.early_stopper = EarlyStopper(patience=5, delta=1e-4, checkpoint_dir=self.save_path)
        self.strategy = self.create_strategy(model, args.toy)
        self.global_model = deepcopy(model)
        
    def fit_config(self, server_round: int) -> Dict[str, int]:
        utils.print_func_and_line()
        return {
            "server_round": server_round,
            "batch_size": 16,
            "local_epochs": 1 if server_round < 2 else 2,
        }

    def evaluate_config(self, server_round: int) -> Dict[str, int]:
        utils.print_func_and_line()
        val_steps = 5 if server_round < 4 else 10
        return {
            "val_steps": val_steps, 
            "server_round": server_round
        }

    def __get_evaluate_fn(self, model: torch.nn.Module, toy: bool):
        utils.print_func_and_line()
        # trainset, _, _ = utils.load_data()
        pascal_voc_partition = PascalVocPartition(args=self.args)
        trainset, testset, publicset = pascal_voc_partition.load_partition(-1)
        n_train = len(testset)
        print(f"n_train: {n_train}")
        if toy:
            valset = torch.utils.data.Subset(testset, range(n_train - 10, n_train))
            publicset = torch.utils.data.Subset(publicset, range(0, 10))
        else:
            valset = torch.utils.data.Subset(testset, range(0, n_train))

        valLoader = DataLoader(valset, batch_size=self.args.batch_size)
        publicLoader = DataLoader(publicset, batch_size=self.args.batch_size, shuffle=False)

        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            utils.print_func_and_line()
            print(f"server_round: {server_round}")
            if server_round == 0 or server_round == 1:
                return None
            # state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
            # model.load_state_dict(state_dict, strict=True)
            ensemble_outputs = parameters
            # print(f"ensemble_outputs shape:", len(ensemble_outputs), ensemble_outputs[0].shape)
            device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
            # 앙상블된 출력을 이용하여 지식증류를 통한 모델 학습
            # result = utils.train_for_distill(model, publicLoader, valLoader, 30, ensemble_outputs, device, self.args)
            # accuracy = result["val_acc"]
            # loss = result["val_loss"]
            
            # is_best_accuracy = self.early_stopper.is_best_accuracy(accuracy)
            # if is_best_accuracy:
            #     filename = f"model_round{server_round}_acc{accuracy:.2f}_loss{loss:.2f}.pth"
            #     self.early_stopper.save_checkpoint(model, server_round, loss, accuracy, filename)

            # if self.early_stopper.counter >= self.early_stopper.patience:
            #     print(f"Early stopping : {self.early_stopper.counter} >= {self.early_stopper.patience}")
            #     # todo : stop server
                
            # if self.experiment is not None:
            #     result = {k.replace("val_", "test_"): v for k, v in result.items()}
            #     self.experiment.log_metrics(result, step=server_round)
                
            # return float(loss), {"accuracy": float(accuracy)}

        return evaluate
    
    def create_strategy(self, model: torch.nn.Module, toy: bool):
        utils.print_func_and_line()
        print("create_strategy")
        return FedMHAD(
            fraction_fit=1,
            fraction_evaluate=1,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_fn=self.__get_evaluate_fn(model, toy),
            on_fit_config_fn=self.fit_config,
            on_evaluate_config_fn=self.evaluate_config,
            initial_parameters=None,
        )

    def start_server(self, port: int, num_rounds: int):
        utils.print_func_and_line()
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
    parser.add_argument("--alpha", type=float, default=1.0, required=False, help="alpha")
    parser.add_argument("--seed", type=int, default=1, required=False, help="seed")
    parser.add_argument("--num_rounds", type=int, default=100, required=False, help="Number of rounds to run. Default: 100")
    parser.add_argument("--dataset", type=str, default="pascal_voc", required=False, help="Dataset to use. Default: pascal_voc")
    parser.add_argument("--num_classes", type=int, default=20, required=False, help="Number of classes. Default: 10")
    parser.add_argument("--N_parties", type=int, default=5, required=False, help="Number of clients to use. Default: 10")
    parser.add_argument("--task", type=str, default="multilabel", required=False, help="Task to run. Default: multilabel")
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
    experiment.set_name(f"global_({args.port})_lr_{args.learning_rate}_bs_{args.batch_size}_rd_{args.num_rounds}")
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
    custom_server = CustomServerManager(model, args, experiment)
    custom_server.start_server(args.port, args.num_rounds)
    
    if experiment is not None:
        experiment.end()

if __name__ == "__main__":
    main()



