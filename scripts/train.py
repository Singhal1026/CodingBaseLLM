# Implement scripts/train.py:
# Load config
# Prepare dataset and dataloader
# Initialize model
# Set up optimizer and loss
# Run training loop (save checkpoints, log metrics)


import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from data.dataset import TextDataset
from model.transformer import GPTModel
from training.trainer import Trainer
from training.callbacks import ModelCheckpoint, EarlyStopping
# from model.utils import Config

# config = Config().params

def load_config(config_path: str) -> dict:
    print(f"*  Loading config from {config_path}...")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("*  Config loaded successfully!")
    return config



def device_check(config: dict) -> str:
    device = config.get('device', 'auto')
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def create_dataloader(config: dict):
    data_config = config['data']
    data_path = data_config.get('dataset_path', 'data/the-verdict.txt')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    split_ratio = data_config.get('train_ratio', 0.9)
    split_index = int(len(text_data) * split_ratio)

    train_data = text_data[:split_index]
    val_data = text_data[split_index:]

    batch_size = data_config.get('batch_size', 32)
    stride = data_config.get('stride', 256)
    seq_len = data_config.get('seq_len', 256)

    train_dataset = TextDataset(train_data, seq_len, stride)
    val_dataset = TextDataset(val_data, seq_len, stride)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    print("Data Load Summary:")
    print("Train Loader: Batches =", len(train_loader), ", Samples =", len(train_dataset))
    print("Val Loader: Batches =", len(val_loader), ", Samples =", len(val_dataset))
    return train_loader, val_loader


def initialize_model(config: dict):
    model_config = config['model']
    model = GPTModel(config=model_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    # print(f"Model Size: {total_params * 4 / (1024 ** 2):.2f} MB (assuming 32-bit floats)")

    return model


def setup_optimizer(config: dict, model: nn.Module):
    training_config = config['training']

    optimizer_name = training_config.get('optimizer', 'adam').lower()
    lr = training_config.get('learning_rate', 1e-4)
    weight_decay = training_config.get('weight_decay', 0.01)

    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"\n  Optimizer Configuration:")
    print(f"   Type: {optimizer_name.upper()}")
    print(f"   Learning rate: {lr}")
    print(f"   Weight decay: {weight_decay}")
    
    return optimizer


def setup_callbacks(config: dict):
    callbacks = []

    training_config = config['training']

    save_dir = training_config.get('save_dir', 'checkpoints')
    save_interval = training_config.get('save_interval', 1)
    save_best_only = training_config.get('save_best_only', True)

    checkpoint_callback = ModelCheckpoint(
        save_dir=save_dir,
        save_interval=save_interval,
        save_best_only=save_best_only,
        verbose=True
    )

    callbacks.append(checkpoint_callback)

    if training_config.get('early_stopping', False):
        patience = training_config.get('patience', 3)
        min_delta = training_config.get('min_delta', 0.0001)

        early_stopping_callback = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            verbose=True
        )
    
        callbacks.append(early_stopping_callback)
    
    return callbacks



def main(config: str = "configs/gpt_small.yaml"):
    
    config = load_config(config)

    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"\n Random seed set to: {seed}")

    device = device_check(config)
    config['device'] = device

    print("Loading data...")
    train_loader, val_loader = create_dataloader(config)
    
    print("Initializing model...")
    model = initialize_model(config=config).to(device)

    print("Setting up optimizer and loss function...")
    optimizer = setup_optimizer(config, model)
    loss_fn = nn.CrossEntropyLoss()
    print("Loss function: CrossEntropyLoss")

    callbacks = setup_callbacks(config)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn, 
        config=config,
        callbacks=callbacks
    )

    epochs = config['training'].get('epochs', 1)
    print(f"\n Starting training for {epochs} epochs on {device.upper()}")
    trainer.train(train_loader, val_loader, epochs=epochs)
    print("Training completed!")

if __name__ == "__main__":
    # config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gpt_small.yaml"
    config_path = "configs/gpt_small.yaml"
    main(config=config_path)