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


def top_p_sampling(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    probs = F.softmax(sorted_logits, dim=-1)

    cumulative_probs = torch.cumsum(probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p

    # be sure to keep at least one token
    sorted_indices_to_remove[0] = False
    
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    
    probs = F.softmax(sorted_logits, dim=-1)
    
    sampled_index = torch.multinomial(probs, num_samples=1)
    
    token_id = sorted_indices[sampled_index]
    
    return token_id

@torch.no_grad()
def generate_text(
    model: GPTModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    sampling_strategy: str = 'top_k',
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = 'cpu'
)-> str:
    model.eval()

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    generated = input_tensor

    for _ in range(max_length):

        end_token_id = tokenizer.encode("<|endoftext|>")[0] if hasattr(tokenizer, 'encode') else None

        if generated.size(1) > model.config['max_seq_len']:
            input_tensor = generated[:, -model.config['max_seq_len']:]
        else:
            input_tensor = generated

        outputs = model(input_tensor)

        next_token_logits = outputs[:, -1, :] / temperature

        # Sample next token based on strategy
        if sampling_strategy == 'greedy':
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        elif sampling_strategy == 'top_k':
            next_token = top_k_sampling(next_token_logits, k=top_k)
        elif sampling_strategy == 'top_p':
            next_token = top_p_sampling(next_token_logits, p=top_p)
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Stop if we generate an end token
        if end_token_id is not None and next_token.item() == end_token_id:
            break
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated[0].tolist())
    
    return generated_text
            

def main(config = "configs/gpt_small.yaml"):
    pass





if __name__ == "__main__":
    config = "configs/gpt_small.yaml"
    main(config = config)
    



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