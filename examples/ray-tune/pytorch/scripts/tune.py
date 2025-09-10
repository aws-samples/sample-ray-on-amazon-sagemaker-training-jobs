import os
import tempfile
import sys
from argparse import ArgumentParser

import logging
from model import Net
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from filelock import FileLock
from torch.utils.data import random_split

from ray import train, tune
from ray.train import RunConfig
from ray.tune.schedulers import ASHAScheduler


def parse_args():
    """Parse command line arguments for hyperparameters."""
    parser = ArgumentParser()

    parser.add_argument(
        "--max_epochs", type=int, default=None, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--smoke_test", action="store_true", help="Run smoke test with fake data"
    )

    # Parse only the arguments we care about and ignore the rest
    args, unknown = parser.parse_known_args()

    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    return args


# Configure logging to prevent duplicates in distributed environments
def setup_logging():
    """Set up logging configuration to prevent duplicate messages in Ray workers."""
    logger = logging.getLogger(__name__)

    # Prevent duplicate handlers in distributed environments
    if logger.handlers or hasattr(logger, "_configured"):
        return logger

    # Only configure logging once per process
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],  # Only use stdout, not both stdout and stderr
        force=True,  # Override any existing configuration
    )

    # Mark as configured and prevent propagation to avoid duplicates
    logger._configured = True
    logger.propagate = False

    return logger


logger = setup_logging()


def load_data(data_dir="./data"):
    """Create dataloaders for normalized CIFAR10 training/test subsets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )

        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform
        )

    return trainset, testset


def create_dataloaders(trainset, batch_size, num_workers=8):
    """Create train/val splits and dataloaders."""
    train_size = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [train_size, len(trainset) - train_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader


def load_test_data():
    # Load fake data for running a quick smoke-test.
    trainset = torchvision.datasets.FakeData(
        128, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    testset = torchvision.datasets.FakeData(
        16, (3, 32, 32), num_classes=10, transform=transforms.ToTensor()
    )
    return trainset, testset


def train_cifar(config):
    net = Net(config["l1"], config["l2"])
    device = config["device"]
    if device == "cuda":
        net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-5
    )

    # Load existing checkpoint through `get_checkpoint()` API.
    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # Data setup
    if config["smoke_test"]:
        trainset, _ = load_test_data()
    else:
        trainset, _ = load_data()
    train_loader, val_loader = create_dataloaders(
        trainset, config["batch_size"], num_workers=0 if config["smoke_test"] else 8
    )

    for epoch in range(
        config["max_num_epochs"]
    ):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()  # reset gradients
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation
        net.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # Report metrics
        metrics = {
            "loss": val_loss / len(val_loader),
            "accuracy": correct / total,
        }

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file-like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)
    print("Finished Training!")


def test_best_model(best_result, smoke_test=False):
    best_trained_model = Net(best_result.config["l1"], best_result.config["l2"])
    device = best_result.config["device"]
    if device == "cuda":
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    checkpoint_path = os.path.join(
        best_result.checkpoint.to_directory(), "checkpoint.pt"
    )

    model_state, _optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    if smoke_test:
        _trainset, testset = load_test_data()
    else:
        _trainset, testset = load_data()

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Best trial test set accuracy: {correct / total}")

    # Save the best model
    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_trained_model.state_dict(), os.path.join(model_dir, "model.pth"))
    print(f"Model saved to {model_dir}/model.pth")


if __name__ == "__main__":
    args = parse_args()

    config = {
        "l1": (tune.sample_from(lambda _: 2 ** np.random.randint(2, 9))),
        "l2": (tune.sample_from(lambda _: 2 ** np.random.randint(2, 9))),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "smoke_test": args.smoke_test,
        "num_trials": 10 if not args.smoke_test else 2,
        "max_num_epochs": (
            args.max_epochs if args.max_epochs else (10 if not args.smoke_test else 2)
        ),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_cifar),
            resources={"cpu": 2, "gpu": 1 if torch.cuda.is_available() else 0},
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=config["num_trials"],
        ),
        run_config=RunConfig(
            storage_path="/opt/ml/output/data"  # Use SageMaker's shared output directory
        ),
        param_space=config,
    )
    results = tuner.fit()

    best_result = results.get_best_result("loss", "min")

    logger.info(f"Best trial config: {best_result.config}")
    logger.info(f"Best trial final validation loss: {best_result.metrics['loss']}")
    logger.info(
        f"Best trial final validation accuracy: {best_result.metrics['accuracy']}"
    )

    test_best_model(best_result, smoke_test=args.smoke_test)
