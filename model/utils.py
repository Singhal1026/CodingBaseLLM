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
        self.shift = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.scale * x_norm + self.shift
    

class GELU(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * 0.5 * (1.0 + torch.tanh(torch.sqrt(2.0 / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        return self.layers(x)
    

        


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