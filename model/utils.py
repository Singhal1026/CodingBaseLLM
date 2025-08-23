import torch
import torch.nn as nn
import os
import yaml


class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'gpt_small.yaml')
            with open(config_path, 'r') as f:
                cls._instance.params = yaml.safe_load(f)
        return cls._instance



class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.parameter(torch.zeros(d_model))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(sim=-1, keepdim=True, unbiased=False)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.scale * x_norm + self.shift
    

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init()

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        return self.layers(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, din, dout, num_heads):
        super().__init__()

        self.din = din
        self.dout = dout
        self.num_heads = num_heads
        self.head_dim = dout // num_heads

        self.Wq = nn.parameter(torch.randn(din, dout))
        self.Wk = nn.parameter(torch.randn(din, dout))
        self.Wv = nn.parameter(torch.randn(din, dout))


    def forward(self, x):
        # x --> (batch_size, seq_le, d_model(din)) 
        batch_size, seq_len, _ = x.shape
        q = self.Wq @ x          # (batch_size, seq_len, d_model)
        k = self.Wk @ x
        v = self.Wv @ x

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  

        attn_scores = torch.matmul(q, k.transpose(-2, -1))/torch.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        


# class DropPath(nn.Module):
#     def __init__(self, drop_prob=None):
#         super().__init__()
#         self.drop_prob = drop_prob

#     def forward(self, x):
#         if self.drop_prob == 0.0 or not self.training:
#             return x
#         keep_prob = 1 - self.drop_prob
#         random_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
#         random_tensor = random_tensor < keep_prob
#         output = x.div(keep_prob) * random_tensor
#         return output

