import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import os
import inspect
from metrics import compute_mean_average_precision, multi_label_top_margin_k_accuracy, compute_multi_accuracy
import warnings
import unittest

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print_file_and_line = lambda: print(f"file : {os.path.basename(inspect.stack()[1][1])} / line : {inspect.stack()[1][2]}")
print_func_and_line = lambda: print(f"{os.path.basename(inspect.stack()[1].filename)}::{inspect.stack()[1].function}:{inspect.stack()[1].lineno}")

def set_seed(seed):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    
def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = CIFAR10("~/.data", train=True, download=True, transform=transform)
    testset = CIFAR10("~/.data", train=False, download=True, transform=transform)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples

def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(10)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 10)
    n_test = int(num_examples["testset"] / 10)

    # torch.utils.data.Subset : Subset of a dataset at specified indices.
    train_parition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_parition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_parition, test_parition)

def train(net, trainloader, valloader, epochs, device: str = "cpu", args=None):
    print_func_and_line()
    """Train the network on the training set."""
    print("Starting training...")
    
    net.to(device)  # move model to GPU if available
    if args.task == 'singlelabel' : 
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    
    last_layer_name = list(net.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in net.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*10},
    ]
    # if args.optim == 'SGD':
    optimizer = torch.optim.SGD( params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # else:    
        # optimizer = torch.optim.Adam( params= parameters, lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
        
    net.train()
    for i in range(epochs):
        print("Epoch: ", i)
        for images, labels in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if args.task == 'singlelabel' : 
                loss = criterion(net(images), labels)
            else:
                loss = criterion(net(images), labels.float())
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU

    # train_loss, train_acc = test(net, trainloader)
    results1 = test(net, trainloader, args=args)
    results1 = {f"train_{k}": v for k, v in results1.items()}
    # val_loss, val_acc = test(net, valloader)
    results2 = test(net, valloader, args=args)
    results2 = {f"val_{k}": v for k, v in results2.items()}
    results = {**results1, **results2}
    return results

def test(net, testloader, steps: int = None, device: str = "cpu", args=None):
    print_func_and_line()
    """Validate the network on the entire test set."""
    print("Starting evalutation...")
    net.to(device)  # move model to GPU if available
    if args.task == 'singlelabel' : 
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    correct, loss = 0, 0.0
    net.eval()
    m = torch.nn.Sigmoid()
    output_list = []
    target_list = []
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in tqdm(enumerate(testloader)):
            images, targets = images.to(device), targets.to(device)
            outputs = net(images)

            output_list.append(m(outputs).cpu().numpy())
            target_list.append(targets.cpu().numpy())
            
            if args.task == 'singlelabel' :
                loss += criterion(outputs, targets).item()
            else:
                loss += criterion(outputs, targets.float()).item()
            total += outputs.size(0)
            if args.task == 'singlelabel' :
                _, predicted = torch.max(outputs.data, axis=1)
                correct += predicted.eq(targets).sum().item()
            else:
                predicted = torch.sigmoid(outputs) > 0.5
                correct += predicted.eq(targets).all(axis=1).sum().item()
                
            if steps is not None and batch_idx == steps:
                break
    
    output = np.concatenate(output_list, axis=0)
    target = np.concatenate(target_list, axis=0)

    acc, = compute_multi_accuracy(output, target)
    top_k = multi_label_top_margin_k_accuracy(target, output, margin=0)
    mAP, _ = compute_mean_average_precision(target, output)
    acc, top_k, mAP = round(acc, 4), round(top_k, 4), round(mAP, 4)
    print("Accuracy: ", acc, "Top-k: ", top_k, "mAP: ", mAP)
    print("Accuracy: ", correct / total, "Loss: ", loss / total)
            
    loss /= len(testloader.dataset)
    accuracy = correct / len(testloader.dataset)
    print("Accuracy: ", accuracy, "Loss: ", loss)
    net.to("cpu")  # move model back to CPU
    
    return {"loss": loss, "accuracy": accuracy, "acc": acc, "top_k": top_k, "mAP": mAP}


def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)

def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):
    """Loads pretrained efficientnet model from torch hub. Replaces final
    classifying layer if classes is specified.

    Args:
        entrypoint: EfficientNet model to download.
                    For supported entrypoints, please refer
                    https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
        classes: Number of classes in final classifying layer. Leave as None to get the downloaded
                 model untouched.
    Returns:
        EfficientNet Model

    Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
    """
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
    )

    if classes is not None:
        replace_classifying_layer(efficientnet, classes)
    return efficientnet

def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def get_model_output(model, public_loader, device: str = "cpu"):
    """Returns a model's output on the public_loader."""
    model.to(device)
    model.eval()
    m = torch.nn.Sigmoid()
    outputs = []
    with torch.no_grad():
        for images, _ in public_loader:
            images = images.to(device)
             # move the model to the same device as the input tensor
            outputs.append(m(model(images)).cpu().numpy())
    outputs = np.concatenate(outputs)
    return outputs

def train_for_distill(net, public_loader, valloader, epochs, ensemble_outputs, device: str = "cpu", args=None):
    print_func_and_line()
    """Train the network on the training set."""
    print("Starting training...")
    
    net.to(device)  # move model to GPU if available
    if args.task == 'singlelabel' : 
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = torch.nn.MultiLabelSoftMarginLoss().to(device)
    
    last_layer_name = list(net.named_children())[-1][0]
    parameters = [
        {'params': [p for n, p in net.named_parameters() if last_layer_name not in n], 'lr': args.learning_rate},
        {'params': [p for n, p in net.named_parameters() if last_layer_name in n], 'lr': args.learning_rate*10},
    ]
    # if args.optim == 'SGD':
    optimizer = torch.optim.SGD( params= parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    # else:    
        # optimizer = torch.optim.Adam( params= parameters, lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    # slice ensemble outputs per batch size
    ensemble_outputs = ensemble_outputs.reshape(public_loader.batch_size, -1, ensemble_outputs.shape[-1])
    net.train()
    for i in range(epochs):
        print("Epoch: ", i)
        for i, (images, _) in enumerate(public_loader):
            images = images.to(device)
            labels = torch.from_numpy(ensemble_outputs[i]).to(device)
            optimizer.zero_grad()
            if args.task == 'singlelabel' : 
                loss = criterion(net(images), labels)
            else:
                loss = criterion(net(images), labels.float())
            loss.backward()
            optimizer.step()

    net.to("cpu")  # move model back to CPU
    # train_loss, train_acc = test(net, public_loader)
    results1 = test(net, public_loader, args=args)
    results1 = {f"train_{k}": v for k, v in results1.items()}
    # val_loss, val_acc = test(net, valloader)
    results2 = test(net, valloader, args=args)
    results2 = {f"val_{k}": v for k, v in results2.items()}
    results = {**results1, **results2}
    return results

if __name__ == '__main__':
    unittest.main()
# %%