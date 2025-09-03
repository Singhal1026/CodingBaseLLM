from typing import List, Union
import tiktoken
import torch
from transformers import GPT2TokenizerFast

class Tokenizer:
    def __init__(self, path="data/gpt2_encoding", offline: bool = False):
        if offline:
            vocab = f"{path}/encoder.json"
            merges = f"{path}/vocab.bpe"
            self.tokenizer = GPT2TokenizerFast(vocab_file=vocab, merges_file=merges, unk_token="")
        else:
            self.tokenizer = tiktoken.get_encoding("gpt2")
    
    # here max_len is used to truncate the output tokens to a maximum length
    def encode(self, text: str, max_len: int = None) -> torch.Tensor:
        tokens = self.tokenizer.encode(text)
        if max_len:
            tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)
        
    def decode(self, token_ids: Union[torch.tensor, List[int]]) -> str:
        if isinstance(token_ids, torch.tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids)

