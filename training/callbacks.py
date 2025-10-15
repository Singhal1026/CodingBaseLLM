import os
import torch
from pathlib import Path
from typing import Optional


class Callback:
    def on_epoch_end(self, epoch: int, model: torch.nn.Module, train_loss: float, val_loss: Optional[float]) -> None:
        pass

    def on_batch_end(self, batch: int, loss: float) -> None:
        pass


class ModelCheckpoint(Callback):
    def __init__(self, save_dir: str = "checkpoints", save_interval: int = 1, save_best_only: bool = True, verbose: bool = True):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf')
        self.verbose = verbose
        self.save_interval = save_interval

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, train_loss: float, val_loss: Optional[float]) -> None:
        
        if epoch % self.save_interval == 0:
            if self.save_best_only and val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, model, train_loss, val_loss)
            else:
                self._save_checkpoint(model, epoch, val_loss, train_loss)

    def _save_checkpoint(self, epoch: int, model: torch.nn.Module, train_loss: float, val_loss: Optional[float]) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss
        }

        if val_loss is not None:
            filename = f"model_epoch_{epoch}_val_loss_{val_loss:.4f}.pt"
            if self.verbose:
                print(f"Validation loss improved. Saving model to {filename}")
        else:
            filename = f"model_epoch_{epoch}_train_loss_{train_loss:.4f}.pt"
            if self.verbose:
                print(f"Saving model to {filename}")
        save_path = Path(self.save_dir) / filename
        torch.save(checkpoint, save_path)

        latest_checkpoint = Path(self.save_dir) / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_checkpoint)


class EarlyStopping(Callback):
    def __init__(self, patience: int = 3, min_delta: float = 0.0001, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_val_loss = float('inf')
        self.counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch: int, model: torch.nn.Module, train_loss: float, val_loss: Optional[float] = None) -> None:
        if val_loss is None:
            return
        
        if self.best_val_loss - val_loss > self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Early stopping triggered. Stopping training.")
        
    def reset(self):
        self.best_val_loss = float('inf')
        self.counter = 0
        self.stop_training = False
        