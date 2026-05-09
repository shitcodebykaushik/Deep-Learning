import torch
import torch.nn as nn
import math

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Single linear for Q, K, V combined (more efficient)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
        # Causal mask (prevents looking at future tokens)
        self.register_buffer("mask", torch.triu(torch.ones(256, 256), diagonal=1).bool())
    
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, T, head_dim)
        
        # Attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:T, :T], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Pre-norm architecture (more stable for small models)
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
            TransformerBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
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
        logits = self.head(x)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# Test
if __name__ == "__main__":
    model = TinyPythonLLM()
    print(f"Parameters: {model.count_parameters()/1e6:.2f}M")
    
    # Test forward pass
    x = torch.randint(0, 3000, (2, 64))  # batch=2, seq=64
    logits = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")  # Should be (2, 64, 3000)