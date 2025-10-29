import sys
import torch
import yaml
from pathlib import Path
import torch.nn.functional as F


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.transformer import GPTModel
from data.tokenizer import Tokenizer 


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path, device):

    model = GPTModel(config=config['model'])

    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def top_k_sampling(logits: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    top_k_logits, top_k_indices = torch.topk(logits, top_k)
    probs = F.softmax(top_k_logits, dim=-1)
    sampled_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
    token_id = top_k_indices[torch.arange(logits.size(0)), sampled_indices]
    return token_id

def generate_next_tokens(model, max_new_tokens, idx, seq_len, temperature=1.0, top_k=None):
    # idx (batch_size, seq_len) 
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -seq_len:]
        with torch.no_grad():
            logits = model(idx_cond)             # (batch_size, seq_len, vocab_size)
        logits = logits[:, -1, :] / temperature  # (batch_size, vocab_size)
        if top_k is not None:
            top_k_logits, _ = torch.topk(logits, top_k)
            logits[logits < top_k_logits[:, [-1]]] = -float('Inf')
        
        probs = torch.nn.functional.softmax(logits, dim=-1)  # (batch_size, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        idx = torch.cat((idx, next_token), dim=1) # (batch_size, seq_len+1)
    
    return idx