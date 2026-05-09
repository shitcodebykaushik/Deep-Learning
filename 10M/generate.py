import torch
import torch.nn as nn
import math
from tokenizers import Tokenizer

# ============ MODEL (same architecture as training) ============

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=128):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.register_buffer("mask", torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout, max_len=128):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads, max_len)
        self.ffn = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x

class TinyPythonLLM(nn.Module):
    def __init__(self, vocab_size=3000, d_model=256, num_layers=6, num_heads=8, ffn_dim=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, ffn_dim, dropout, max_len) for _ in range(num_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=0.8, top_k=20):
        """Generate tokens autoregressively."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -128:]  # Crop to max sequence length
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============ GENERATION ============

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    model = TinyPythonLLM(vocab_size=3000, d_model=256, num_layers=6, num_heads=8, ffn_dim=1024, max_len=128, dropout=0.0)
    
    checkpoint = torch.load("checkpoint_epoch3.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    
    prompts = [
        "def factorial(n):",
        "def hello_world():",
        "import os",
        "class Person:",
        "for i in range(10):",
    ]
    
    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        
        encoded = tokenizer.encode(prompt)
        idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        
        generated = model.generate(idx, max_new_tokens=50, temperature=0.8, top_k=20)
        decoded = tokenizer.decode(generated[0].tolist())
        
        print(decoded)
        print()

if __name__ == "__main__":
    main()