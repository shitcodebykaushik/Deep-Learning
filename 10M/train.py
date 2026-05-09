import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import time

# ============ MODEL (same as before) ============

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=256):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.register_buffer("mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())
    
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout, max_len=256):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, max_len)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class TinyPythonLLM(nn.Module):
    def __init__(self, vocab_size=3000, d_model=256, num_layers=6, 
                 num_heads=8, ffn_dim=1024, max_len=256, dropout=0.1):
        super().__init__()
        
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim, dropout, max_len)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_final(x)
        return self.head(x)


# ============ DATASET ============

class TokenDataset(Dataset):
    def __init__(self, token_file, seq_len=128):
        self.tokens = np.load(token_file)
        self.seq_len = seq_len
        self.num_samples = len(self.tokens) - seq_len
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============ TRAINING ============

def train():
    # Hyperparameters
    SEQ_LEN = 128
    BATCH_SIZE = 8
    NUM_EPOCHS = 3
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 500
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data
    dataset = TokenDataset("python_train_tokens.npy", seq_len=SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    # Model
    model = TinyPythonLLM(vocab_size=3000, d_model=256, num_layers=6,
                          num_heads=8, ffn_dim=1024, max_len=SEQ_LEN, dropout=0.1)
    model = model.to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler (cosine with warmup)
    total_steps = len(loader) * NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - WARMUP_STEPS)
    
    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            
            # Forward
            logits = model(x)
            
            # Compute loss (flatten for CrossEntropy)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # LR warmup
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                scheduler.step()
            
            epoch_loss += loss.item()
            step += 1
            
            # Print progress
            if batch_idx % 200 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Step {step} | "
                      f"Batch {batch_idx}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {elapsed:.1f}s")
        
        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch+1} complete. Average loss: {avg_loss:.4f}\n")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, f"checkpoint_epoch{epoch+1}.pt")
        print(f"Saved: checkpoint_epoch{epoch+1}.pt")
    
    print("Training complete!")
    
    # Save final model
    torch.save(model.state_dict(), "tiny_python_llm_final.pt")
    print("Saved: tiny_python_llm_final.pt")


if __name__ == "__main__":
    train()