from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
from typing import Optional
import warnings
import utils

warnings.filterwarnings("ignore")

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        device: str,
        validation_split: int = 0.1,
        experiment: Optional[Experiment] = None,
        args: Optional[argparse.Namespace] = None,
    ):
        self.device = device
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split
        self.experiment = experiment
        self.args = args

    def set_parameters(self, parameters):
        """Loads a efficientnet model and replaces it parameters with the ones
        given."""
        model = utils.load_efficientnet(classes=10)
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

        results = utils.train(model, trainLoader, valLoader, epochs, self.device, self.args)
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

        loss, accuracy = utils.test(model, testloader, steps, self.device)
        self.experiment.log_metrics({"test_loss": loss, "test_accuracy": accuracy}, step=server_round)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run(device: str = "cpu"
                   , experiment: Optional[Experiment] = None
                   , args: Optional[argparse.Namespace] = None) -> None:
    """Weak tests to check whether all client methods are working as
    expected."""

    model = utils.load_efficientnet(classes=10)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(trainset, testset, device, experiment= experiment, args= args)
    client.fit(
        utils.get_model_params(model),
        {"batch_size": 16, "local_epochs": 1},
    )

    client.evaluate(utils.get_model_params(model), {"val_steps": 32})
    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    utils.set_seed(42)
    
    experiment = Experiment(
    api_key = "3JenmgUXXmWcKcoRk8Yra0XcD",
    project_name = "test1",
    workspace="neighborheo"
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start Flower server with experiment key.")
    parser.add_argument("--index", type=int, required=True, help="Index of the client")
    parser.add_argument("--experiment_key", type=str, required=True, help="Experiment key")
    parser.add_argument("--toy", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--use_cuda", type=bool, default=False, required=False, help="Set to true to use GPU. Default: False" )
    parser.add_argument("--partition", type=int, default=0, choices=range(0, 10), required=False, help="Specifies the artificial data partition of CIFAR10 to be used. Picks partition 0 by default" )
    parser.add_argument("--dry", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--port", type=int, default=8080, required=False, help="Port to use for the server. Default: 8080")
    parser.add_argument("--learning_rate", type=float, default=0.1, required=False, help="Learning rate. Default: 0.1")
    parser.add_argument("--momentum", type=float, default=0.9, required=False, help="Momentum. Default: 0.9")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay. Default: 1e-4")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size. Default: 32")
    
    args = parser.parse_args()
    print("Experiment key:", args.experiment_key, "port:", args.port)

    # Report multiple hyperparameters using a dictionary:
    experiment.log_parameters(args)
    experiment.set_name(f"client_{args.index}_lr_{args.learning_rate}_bs_{args.batch_size}_p{args.port}")

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu"
    )

    if args.dry:
        client_dry_run(device, experiment, args)
    else:
        # Load a subset of CIFAR-10 to simulate the local data partition
        trainset, testset = utils.load_partition(args.index)

        if args.toy:
            trainset = torch.utils.data.Subset(trainset, range(10))
            testset = torch.utils.data.Subset(testset, range(10))

        # Start Flower client
        client = CifarClient(trainset, testset, device, 0.1, experiment, args)
        fl.client.start_numpy_client(server_address=f"0.0.0.0:{args.port}", client=client)

    experiment.end()

if __name__ == "__main__":
    main()
