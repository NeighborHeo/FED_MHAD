from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict

from typing import Optional
import warnings
import utils
from model import vit_tiny_patch16_224
from dataset import PascalVocPartition, Cifar10Partition
from early_stopper import EarlyStopper

warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env_comet'))

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model

class CustomClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        validation_split: int = 0.1,
        experiment: Optional[Experiment] = None,
        args: Optional[argparse.Namespace] = None,
    ):
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.experiment = experiment
        self.args = args
        self.save_path = f"checkpoints/{args.port}/client_{args.index}_best_models"
        self.early_stopper = EarlyStopper(patience=5, delta=1e-4, checkpoint_dir=self.save_path)
        

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = vit_tiny_patch16_224(pretrained=True, num_classes=self.args.num_classes)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # save file 
        model.load_state_dict(state_dict, strict=True)
        return model

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        server_round: int = config["server_round"]

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        results = utils.train(model, trainLoader, valLoader, epochs, device, self.args)
        
        accuracy = results["val_accuracy"]
        loss = results["val_loss"]
        is_best_accuracy = self.early_stopper.is_best_accuracy(accuracy)
        if is_best_accuracy:
            filename = f"model_round{server_round}_acc{accuracy:.2f}_loss{loss:.2f}.pth"
            self.early_stopper.save_checkpoint(model, server_round, loss, accuracy, filename)

        if self.early_stopper.counter >= self.early_stopper.patience:
            print(f"Early stopping : {self.early_stopper.counter} >= {self.early_stopper.patience}")
            # todo : stop server
        
        if self.experiment is not None:
            self.experiment.log_metrics(results, step=server_round)
        
        parameters_prime = utils.get_model_params(model)
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        model = self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]
        server_round: int = config["server_round"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=16)

        device = torch.device("cuda:0" if torch.cuda.is_available() and self.args.use_cuda else "cpu")
        result = utils.test(model, testloader, steps, device, self.args)
        accuracy = result["acc"]
        loss = result["loss"]
        result = {f"test_" + k: v for k, v in result.items()}
        
        self.experiment.log_metrics(result, step=server_round)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(experiment: Optional[Experiment] = None
                   , args: Optional[argparse.Namespace] = None) -> None:
    """Weak tests to check whether all client methods are working as
    expected."""
    
    model = vit_tiny_patch16_224(pretrained=True, num_classes=args.num_classes)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print("Using device:", device)
    client = CustomClient(trainset, testset, device, experiment= experiment, args= args)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})
    print("Dry Run Successful")

def init_argurments():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start Flower server with experiment key.")
    parser.add_argument("--index", type=int, required=True, help="Index of the client")
    parser.add_argument("--experiment_key", type=str, required=True, help="Experiment key")
    parser.add_argument("--toy", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--use_cuda", type=bool, default=True, required=False, help="Set to true to use GPU. Default: False" )
    parser.add_argument("--partition", type=int, default=0, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. Picks partition 0 by default" )
    parser.add_argument("--dry", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--port", type=int, default=8080, required=False, help="Port to use for the server. Default: 8080")
    parser.add_argument("--learning_rate", type=float, default=0.00002, required=False, help="Learning rate. Default: 0.1")
    parser.add_argument("--momentum", type=float, default=0.9, required=False, help="Momentum. Default: 0.9")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay. Default: 1e-4")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size. Default: 32")
    parser.add_argument("--datapath", type=str, default="~/.data/", required=False, help="dataset path")
    parser.add_argument("--alpha", type=float, default=0.5, required=False, help="alpha")
    parser.add_argument("--seed", type=int, default=1, required=False, help="seed")
    parser.add_argument("--num_rounds", type=int, default=100, required=False, help="Number of rounds to run. Default: 100")
    parser.add_argument("--dataset", type=str, default="cifar10", required=False, help="Dataset to use. Default: pascal_voc")
    parser.add_argument("--num_classes", type=int, default=10, required=False, help="Number of classes. Default: 10")
    parser.add_argument("--N_parties", type=int, default=20, required=False, help="Number of clients to use. Default: 10")
    parser.add_argument("--task", type=str, default="singlelabel", required=False, help="Task to run. Default: multilabel")
    parser.add_argument("--noisy", type=float, default=0.0, required=False, help="Percentage of noisy data. Default: 0.0")
    
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
    experiment.set_name(f"client_{args.index}_({args.port})_lr_{args.learning_rate}_bs_{args.batch_size}_ap_{args.alpha}_ns_{args.noisy}")
    return experiment

def main() -> None:
    # Parse command line argument `partition`
    utils.set_seed(42)
    
    args = init_argurments()
    experiment = init_comet_experiment(args)

    if args.dry:
        client_dry_run(experiment, args)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        # trainset, testset = utils.load_partition(args.index)
        
        if args.dataset == "cifar10":
            cifar10_partition = Cifar10Partition(args)
            trainset, testset = cifar10_partition.load_partition(args.index)
        else:
            pascal_voc_partition = PascalVocPartition(args)
            trainset, testset = pascal_voc_partition.load_partition(args.index)
            
        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        client = CustomClient(trainset, testset, 0.1, experiment, args)
        fl.client.start_numpy_client(server_address=f"0.0.0.0:{args.port}", client=client)

    experiment.end()

if __name__ == "__main__":
    main()
