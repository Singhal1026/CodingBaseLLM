import torch
from torch.utils.data import Dataset
from typing import List, Union, Optional
from .tokenizer import Tokenizer


class TextDataset(Dataset):
    def __init__(self, text: Union[str, List[str]], context_length: int, stride: Optional[int]=1):
        super().__init__()
        self.tokenizer = Tokenizer(offline=True)
        self.context_length = context_length
        self.stride = stride

        if isinstance(text, list):
            text = " ".join(text)

        self.tokens = self.tokenizer.encode(text).tolist()

        if len(self.tokens) < context_length + 1:
            padding_len = context_length + 1 - len(self.tokens)
            self.tokens += [self.tokenizer.eos_token_id] * (padding_len)
            self.num_samples = 1
        else:
            # trunctate extra tokens 
            max_valid_length = ((len(self.tokens) - self.context_length - 1) // self.stride) * self.stride + self.context_length + 1
            self.tokens = self.tokens[:max_valid_length]
            
            self.num_samples = (len(self.tokens) - self.context_length) // self.stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        input_ids = torch.tensor(self.tokens[start:end])
        target_ids = torch.tensor(self.tokens[start+1:end+1])

        return input_ids, target_ids
