import sys
import os
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir))
print(sys.path)


from model.transformer import GPTModel
from data.dataset import TextDataset


def load_config(config_path):
    print(f"📄 Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: dict, device: str):
    model_config = config['model']

    model = GPTModel(
        vocab_size = model_config.get('vocab_size', 50257),
        n_layers = model_config.get('n_layers', 12),
        n_heads = model_config.get('n_heads', 12),
        d_model = model_config.get('d_model', 768),
        d_ff = model_config.get('d_ff', 3072),
        seq_len = model_config.get('seq_len', 256),
        dropout = model_config.get('dropout', 0.1)
    )

    model_path = config['evaluation'].get('model_checkpoint', 'checkpoints/latest_checkpoint.pt')
    print(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()        

    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")
    if 'train_loss' in checkpoint:
        print(f"Final train loss: {checkpoint['train_loss']:.4f}")
    if 'val_loss' in checkpoint and checkpoint['val_loss']:
        print(f"Final val loss: {checkpoint['val_loss']:.4f}")

    return model


@torch.no_grad()
def evaluate_model(model, dataloader, loss_fn, device):

    model.eval()

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0

    for input_ids, target_ids in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
        logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)

        target_ids = target_ids.view(-1)  # (batch_size * seq_len)

        loss = loss_fn(logits, target_ids)
        batch_size, seq_len = input_ids.size()
        num_tokens = batch_size * seq_len

        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        total_batches += 1

    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    return avg_loss, perplexity
