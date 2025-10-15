import torch
from torch.utils.data import DataLoader
from model.transformer import GPTModel
from model.utils import Config
from data.tokenizer import Tokenizer
from data.dataset import TextDataset

def test_tokenization():
    config = Config().params
    tokenizer = Tokenizer()
    
    # Test text
    text = "Hello, this is a test sentence."
    # with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
    #     text = f.read()
    #     text = text[:1000]  # Use only the first 1000 characters for testing
    
    # Encode
    tokens = tokenizer.encode(text, max_length=config['model']['seq_len'])
    print(f"Token IDs: {tokens}")
    print(f"Token shape: {tokens.shape}")
    
    # Decode
    decoded = tokenizer.decode(tokens[0])  # First batch
    print(f"Decoded text: {decoded}")


def load_data():
    config = Config().params
    
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        text = f.read()
        text = text[:5000]  # Use only the first 1000 characters for testing
    
    # print(text)
    # Encode

    data = TextDataset(text, context_length = config['model']['seq_len'])
    loader = DataLoader(data, batch_size=2, shuffle=False)
    for batch in loader:
        tokens, targets = batch
        # print(f"Batch token IDs: {tokens}")
        # print(f"Batch target IDs: {targets}")
        print(f"Token shape: {tokens.shape}, Target shape: {targets.shape}")


def test_model():
    
    config = Config().params
    model_config = config['model']

    # dummy input: 
    batch_size, seq_len = (2, 5)
    # below line creates a tensor of shape (batch_size, seq_len) with random integers in the range [0, vocab_size)
    dummy_input = torch.randint(0, model_config['vocab_size'], (batch_size, seq_len))

    model = GPTModel()
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")  # should be (batch_size, seq_len, vocab_size)


if __name__ == "__main__":
    load_data()
    # test_model()