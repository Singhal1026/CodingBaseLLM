import torch
import torch.nn as nn
# from .utils import Config
from .utils import LayerNorm, GELU, FFN, Config


config = Config().params
model_config = config['model']


# class TransformerBlock(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.d_model = model_config['d_model']
        

