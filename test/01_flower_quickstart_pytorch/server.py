import argparse
from typing import List, Tuple, Dict, Optional, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics

import utils

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

# utils.set_seed(42)

experiment = Experiment(
  api_key = "3JenmgUXXmWcKcoRk8Yra0XcD",
  project_name = "test1",
  workspace="neighborheo"
)

# Initialize and train your model
# model = TheModelClass()
# train(model)

# Seamlessly log your Pytorch model
# log_model(experiment, model, model_name="TheModel")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Start Flower server with experiment key.")
parser.add_argument("--experiment_key", type=str, required=True, help="Experiment key")
parser.add_argument("--toy", type=bool, default=False, required=False, help="Set to true to use only 10 datasamples for validation. Useful for testing purposes. Default: False" )

args = parser.parse_args()
print("Experiment key:", args.experiment_key)
args.learning_rate = 0.5
args.steps = 100000
args.batch_size = 50

# Report multiple hyperparameters using a dictionary:
experiment.log_parameters(args)
experiment.set_name(f"srv_lr_{args.learning_rate}_stps_{args.steps}_bch_{args.batch_size}")


def get_evaluate_fn(model: torch.nn.Module):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = utils.load_data()

    n_train = len(trainset)
    valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
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
        experiment.log_metric("loss", loss)
        experiment.log_metric("accuracy", accuracy)
        return loss, {"accuracy": accuracy}

    return evaluate

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

model = utils.load_efficientnet(classes=10)

# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average, evaluate_fn=get_evaluate_fn(model, args.toy))

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

experiment.end()
