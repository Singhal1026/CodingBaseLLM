import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import GPTModel

from tqdm import tqdm
from typing import Optional, Union


class Trainer:
    def __init__(self, model: GPTModel, optimizer: torch.optim.Optimizer, config: dict, loss_fn = None, callbacks = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.config = config
        self.callbacks = callbacks if callbacks is not None else []
        self.device = config.get("device", "cpu") 
        self.model.to(self.device)


    def train(self, train_loader : DataLoader, val_loader: Optional[DataLoader] = None, epochs: Optional[int] = 1) -> None:
        
        epochs = epochs or self.config['epochs']

        for epoch in range(1, epochs + 1):
            epoch_loss = self._train_epoch(train_loader, epoch)
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self.model, epoch_loss, val_loss if val_loader is not None else None)
            

    def _train_epoch(self, train_loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        for batch in loop:
            X, y = [item.to(self.device) for item in batch]
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.loss_fn(outputs.flatten(0, 1), y.flatten())
            loss.backward()               # Backpropagation
            self.optimizer.step()         # Update weights
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # average batch loss
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        loop = tqdm(val_loader, desc="Validating")
        for batch in loop:
            X, y = [item.to(self.device) for item in batch]
            outputs = self.model(X)
            loss = self.loss_fn(outputs.flatten(0, 1), y.flatten())
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # average batch loss
        avg_loss = total_loss / len(val_loader)
        return avg_loss
