import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, context_length, stride=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride

        self.tokens = tokenizer.encode(text, add_special_tokens=False)
        self.num_samples = (len(self.tokens) - context_length) // stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.context_length
        input_ids = torch.tensor(self.tokens[start:end])
        target_ids = torch.tensor(self.tokens[start+1:end+1])
        return input_ids, target_ids
