import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from core.pipelines._config import METRICS_DIR
from core.pipelines._base import Pipeline_Config
from core.pipelines._utils import classifier_train_step, classifier_val_step, classifier_test_step, save_model

class Classifier_Pipeline:

    def __init__(self, model: torch.nn.Module, config: Pipeline_Config):

        self.model = model
        self.config = config

    def train(self,
        train_data: DataLoader,
        val_data: DataLoader,
        add_to_name: str = ""
    ) -> tuple[float]:
        
        name = f"{self.model.__class__.__name__}_{time.time()}" + add_to_name
        writer: SummaryWriter = SummaryWriter(os.path.join(METRICS_DIR, name))

        model: torch.Module = self.model.to(self.config.device)

        loss = self.config.loss_class()
        optimizer: torch.optim.Optimizer = self.config.optimizer_class(
            model.parameters(), **self.config.optimizer_kwargs
        )
        scheduler = self.config.scheduler_class(
            optimizer, **self.config.scheduler_kwargs
        )

        for epoch in tqdm(range(self.config.epochs)):

            train_mean_accuracy = classifier_train_step(
                model, train_data, loss, optimizer, epoch, self.config.device, writer
            )

            val_mean_accuracy = classifier_val_step(
                model, val_data, loss, epoch, self.config.device, writer
            )

            print(
                f"\nTrain and Val. accuracies in epoch {epoch}, lr {scheduler.get_last_lr()}:",
                (round(train_mean_accuracy, 4), round(val_mean_accuracy, 4)),
            )

            scheduler.step()

        save_model(model, name)

        return (train_mean_accuracy, val_mean_accuracy,)
    
    def evaluate(
        self,
        test_data: DataLoader
    ) -> float:
        
        model: torch.Module = self.model.to(self.config.device)
        accuracy: float = classifier_test_step(model, test_data, self.config.device)

        return accuracy
