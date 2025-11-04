import torch
import torch.nn as nn
from .utils import LayerNorm, GELU, FFN


# config = Config().params
# model_config = config['model']


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len = x.shape
        # Explaination for the line below:
        # torch.arange(seq_len) creates a tensor of shape (seq_len,)
        # unsqueeze(0) adds a new dimension at the front, making it (1, seq_len)
        # expand(batch_size, -1) expands this tensor to (batch_size, seq_len) by repeating the sequence for each batch
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_embd = self.token_embedding(x)
        pos_embd = self.position_embedding(pos_ids)
        return self.dropout(token_embd + pos_embd)
    

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.
    Args:
        din (int): Input dimension
        dout (int): Output dimension
        context_len (int): Maximum sequence length
        dropout (float): Dropout probability
        num_heads (int): Number of attention heads
        qkv_bias (bool, optional): Use bias in QKV linear layers. Defaults to False.
    """
    def __init__(self, din: int, dout: int, context_len: int, dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()

        self.din = din
        self.dout = dout
        self.num_heads = num_heads

        assert (dout % num_heads == 0), "d_out must be divisible by num_heads"

        self.head_dim = dout // num_heads

        self.W_query = nn.Linear(din, dout, bias=qkv_bias)
        self.W_key = nn.Linear(din, dout, bias=qkv_bias)
        self.W_value = nn.Linear(din, dout, bias=qkv_bias)
        self.W_out = nn.Linear(dout, dout)

        self.dropout = nn.Dropout(dropout)
        # registering a buffer means that it is not a parameter, but will be saved and loaded with the model
        # it will also be moved to the appropriate device when calling model.to(device)
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1).bool())

    def forward(self, x):
        # x --> (batch_size, seq_len, d_model(din)) 
        batch_size, seq_len, _ = x.shape
        q = self.W_key(x)         # (batch_size, seq_len, d_model)
        k = self.W_query(x)
        v = self.W_value(x)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = attn_scores.masked_fill(self.mask[:seq_len, :seq_len], float('-inf'))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        attn_output = attn_weights @ v # (batch_size, num_heads, seq_len, head_dim)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dout) # (batch_size, seq_len, d_model)
        output = self.W_out(attn_output) # (batch_size, seq_len, d_model)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            config['d_model'],
            config['d_model'],
            config['seq_len'],
            config['dropout_rate'],
            config['n_heads']
        )
        self.ln1 = LayerNorm(config['d_model'])
        self.ln2 = LayerNorm(config['d_model'])
        self.ffn = FFN(config['d_model'], config['dff'])
        self.dropout = nn.Dropout(config['dropout_rate'])

    def forward(self, x):
        shortcut = x
        x = self.ln1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.input_embeddings = InputEmbedding(
            config['vocab_size'],
            config['d_model'],
            config['seq_len'], 
            config['dropout_rate']
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config['n_layers'])
        ])

        self.final_norm = LayerNorm(config['d_model'])
        self.output_projection = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, x):
        x = self.input_embeddings(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits
    