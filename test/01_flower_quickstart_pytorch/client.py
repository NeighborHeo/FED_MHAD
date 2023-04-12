import argparse
import warnings
from collections import OrderedDict


import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

# import utils

from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

# utils.set_seedc(42)

experiment = Experiment(
  api_key = "3JenmgUXXmWcKcoRk8Yra0XcD",
  project_name = "test1",
  workspace="neighborheo"
)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Start Flower server with experiment key.")

parser.add_argument("--index", type=int, required=True, help="Index of the client")
parser.add_argument("--experiment_key", type=str, required=True, help="Experiment key")

args = parser.parse_args()
print("Experiment key:", args.experiment_key)
args.learning_rate = 0.5
args.steps = 100000
args.batch_size = 50


# Report multiple hyperparameters using a dictionary:
experiment.log_parameters(args)
experiment.set_name(f"client_{args.index}_lr_{args.learning_rate}_stps_{args.steps}_bch_{args.batch_size}")

# Initialize and train your model
# model = TheModelClass()
# train(model)

# Seamlessly log your Pytorch model
# log_model(experiment, model, model_name="TheModel")

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for i in range(epochs):
        print("Epoch", i)
        for images, labels in tqdm(trainloader):
            print("images", len(images))
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("~/.data", train=True, download=True, transform=trf)
    testset = CIFAR10("~/.data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# def load_pascal_voc_data():
#     """Load PASCAL VOC (training and test set)."""
#     N_parties = 10
#     priv_data = [None] * N_parties
#     for i in range(N_parties):
#         priv_data[i] = {}
#         path = pathlib.Path(args.datapath).joinpath('PASCAL_VOC_2012', f'N_clients_{args.N_parties}_alpha_{args.alpha:.1f}')
#         party_img = np.load(path.joinpath(f'Party_{i}_X_data.npy'))
#         party_label = np.load(path.joinpath(f'Party_{i}_y_data.npy'))
#         party_img, party_label = filter_images_by_label_type(args.task, party_img, party_label)
#         priv_data[i]['x'] = party_img.copy()
#         priv_data[i]['y'] = party_label.copy()
    
#     print(f"y label : {priv_data[0]['y']}")
#     for i in range(N_parties):
#         print(f'Party_{i} data shape: {priv_data[i]["x"].shape}')
#     print(f'Public data shape: {public_dataset.img.shape}')
#     print(f'Test data shape: {test_dataset.img.shape}')
#     return priv_data, public_dataset, test_dataset

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print("fit")
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("evaluate")
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)

experiment.end()
