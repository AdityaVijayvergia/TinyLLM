print("starting imports")

import torch
import gc
from nanochat.model_untied_weights import GPT, GPTConfig

print("imports done")

def profile_model(depth, max_seq_len, batch_size):
    # Free memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    
    vocab_size = 32768
    model_dim = 1024
    num_heads = 8
    num_kv_heads = 8
    
    model_config_kwargs = dict(
        sequence_len=max_seq_len, 
        vocab_size=vocab_size, 
        n_layers=depth, 
        n_heads=num_heads, 
        hidden_dim=model_dim, 
        n_kv_head=num_kv_heads
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.device("meta"):
        model_config = GPTConfig(**model_config_kwargs)
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    num_params = sum(p.numel() for p in model.parameters())
    
    model_memory = torch.cuda.memory_allocated() / (1024 ** 2)
    
    try:
        # Try to setup optimizers
        optimizers = model.setup_optimizers()
        
        # Fake data
        x = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, max_seq_len), device=device)
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.bfloat16):
            loss = model(x, y)
        
        loss.backward()
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        
        peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    except RuntimeError as e:
        print(f"OOM at depth {depth}: {e}")
        peak_vram = -1
    
    # Free
    del model, model_config
    if 'optimizers' in locals():
        del optimizers
    if 'x' in locals():
        del x
    if 'y' in locals():
        del y
    if 'loss' in locals():
        del loss
    torch.cuda.empty_cache()
    gc.collect()
    
    return num_params, model_memory, peak_vram

def main():
    print("Depth\tParams (M)\tModel VRAM (MB)\tPeak Training VRAM (MB)")
    print("-" * 65)
    
    # Fixed base parameters
    max_seq_len = 64
    batch_size = 50
    
    depths = [2, 4, 8, 12, 16, 20, 24]
    
    for depth in depths:
        params, model_mem, peak_vram = profile_model(depth, max_seq_len, batch_size)
        print(f"{depth}\t{params/1e6:.1f}\t\t{model_mem:.1f}\t\t{peak_vram:.1f}")

if __name__ == "__main__":
    main()
