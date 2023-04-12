from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch
import utils

import warnings
warnings.filterwarnings("ignore")

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "server_round": server_round,
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps, "server_round": server_round}


def get_evaluate_fn(model: torch.nn.Module, toy: bool, experiment: Optional[Experiment] = None):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = utils.load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 5k training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    valLoader = DataLoader(valset, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, valLoader)
        if experiment is not None:
            experiment.log_metric("central_loss", loss, step=server_round)
            experiment.log_metric("central_accuracy", accuracy, step=server_round)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    utils.set_seed(42)

    experiment = Experiment(
        api_key = "3JenmgUXXmWcKcoRk8Yra0XcD",
        project_name = "test1",
        workspace="neighborheo"
    )

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Start Flower server with experiment key.")
    parser.add_argument("--experiment_key", type=str, required=True, help="Experiment key")
    parser.add_argument("--toy", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )
    parser.add_argument("--port", type=int, default=8080, required=False, help="Port to use for the server. Default: 8080")
    parser.add_argument("--rounds", type=int, default=10, required=False, help="Number of rounds to run. Default: 10")
    parser.add_argument("--learning_rate", type=float, default=0.1, required=False, help="Learning rate. Default: 0.1")
    parser.add_argument("--momentum", type=float, default=0.9, required=False, help="Momentum. Default: 0.9")
    parser.add_argument("--weight_decay", type=float, default=1e-4, required=False, help="Weight decay. Default: 1e-4")
    parser.add_argument("--batch_size", type=int, default=32, required=False, help="Batch size. Default: 32")
    
    args = parser.parse_args()
    print("Experiment key:", args.experiment_key, "port:", args.port)

    # Report multiple hyperparameters using a dictionary:
    experiment.log_parameters(args)
    experiment.set_name(f"srv_lr_{args.learning_rate}_lr_{args.learning_rate}_bs_{args.batch_size}_rd_{args.rounds}_p_{args.port}")

    # Parse command line argument `partition`
    model = utils.load_efficientnet(classes=10)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1, #0.2,
        fraction_evaluate=1, #0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2, #10,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )
    
    experiment.end()

if __name__ == "__main__":
    main()



