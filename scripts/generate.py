import torch



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