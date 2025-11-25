import os
import numpy as np
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.jit import RecursiveScriptModule

from core.pipelines._config import MODELS_SAVE_DIR

def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model as a torch.jit.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    if not os.path.isdir(MODELS_SAVE_DIR):
        os.makedirs(MODELS_SAVE_DIR)

    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(os.path.join(MODELS_SAVE_DIR, name + ".pt"))

    return None

def load_model(name: str):

    if name[-2:] != ".pt":
        name += ".pt"
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}")

    return model

def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    This function computes the accuracy.

    Args:
        predictions: predictions tensor. Dimensions:
            [batch, num classes] or [batch].
        targets: targets tensor. Dimensions: [batch, 1] or [batch].

    Returns:
        the accuracy in a tensor of a single element.
    """

    if len(predictions.shape) > 1:

        predictions = torch.argmax(predictions, dim=1)
    
    targets = torch.squeeze(targets)

    return (predictions == targets).float().mean()

def classifier_train_step(
    classifier: torch.nn.Module,
    train_data: DataLoader,
    loss: torch.nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
    writer: Optional[SummaryWriter],
) -> float:
    """
    This function computes the training step for an images classifier.

    Args:
        model: pytorch model.
        train_data: train dataloader.
        loss: loss function.
        optimizer: optimizer object.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.

    Returns:
        Mean accuracy over training epochs
    """

    losses: list[float] = []
    accuracies: list[float] = []

    classifier.train()

    for images, labels in train_data:

        images = images.to(device)
        labels = labels.to(device)

        outputs = classifier(images)
        loss_value = loss(outputs, labels)

        # METRICS
        losses.append(loss_value.item())
        accuracies.append(calculate_accuracy(outputs, labels).item())

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

    if writer is not None:
        writer.add_scalar("train/loss", np.mean(losses), epoch)
        writer.add_scalar("train/accuracy", np.mean(accuracies), epoch)

    return float(np.mean(accuracies))

def classifier_val_step(
    classifier: torch.nn.Module,
    val_data: DataLoader,
    loss: torch.nn.modules.loss._Loss,
    epoch: int,
    device: torch.device,
    writer: Optional[SummaryWriter],
) -> float:
    """
    This function computes the validation step for an images classifier.

    Args:
        model: pytorch model.
        val_data: dataloader of validation data.
        loss: loss function.
        writer: tensorboard writer.
        epoch: epoch number.
        device: device of model.

    Returns:
        Average accuracy during validation epochs
    """

    losses: list[float] = []
    accuracies: list[float] = []

    classifier.eval()

    for images, labels in val_data:
        
        with torch.no_grad():

            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)
            loss_value = loss(outputs, labels)

        # METRICS
        losses.append(loss_value.item())
        accuracies.append(calculate_accuracy(outputs, labels).item())

    if writer is not None:
        writer.add_scalar("val/loss", np.mean(losses), epoch)
        writer.add_scalar("val/accuracy", np.mean(accuracies), epoch)

    return float(np.mean(accuracies))

def classifier_test_step(
    classifier: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function computes the test step for an images classifier.

    Args:
        model: pytorch model.
        val_data: dataloader of test data.
        device: device of model.
        
    Returns:
        Average accuracy during test epochs.
    """

    accuracies: list[float] = []
    
    classifier.eval()

    for images, labels in test_data:

        with torch.no_grad():
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier(images)

        accuracies.append(calculate_accuracy(outputs, labels).item())

    return np.mean(accuracies)