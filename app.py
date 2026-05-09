import torch
import torch.nn as nn
import math
from tokenizers import Tokenizer
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# ============ MODEL (same as before) ============

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
    def __init__(self, vocab_size=3000, d_model=256, num_layers=6, num_heads=8, ffn_dim=1024, max_len=128, dropout=0.0):
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
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -128:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.softmax(logits / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============ LOAD MODEL ============

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = Tokenizer.from_file("10M/tokenizer.json")

model = TinyPythonLLM(vocab_size=3000, d_model=256, num_layers=6, num_heads=8, ffn_dim=1024, max_len=128, dropout=0.0)
checkpoint = torch.load("10M/checkpoint_epoch3.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
print(f"Model loaded. Device: {device}")


# ============ HTML TEMPLATE ============

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tiny Python LLM</title>
    <style>
        body { font-family: monospace; max-width: 800px; margin: 50px auto; padding: 20px; background: #1e1e1e; color: #d4d4d4; }
        h1 { color: #4ec9b0; }
        textarea { width: 100%; height: 120px; background: #252526; color: #d4d4d4; border: 1px solid #454545; padding: 10px; font-size: 14px; }
        button { background: #0e639c; color: white; border: none; padding: 10px 20px; font-size: 16px; cursor: pointer; margin-top: 10px; }
        button:hover { background: #1177bb; }
        #output { background: #252526; border: 1px solid #454545; padding: 15px; margin-top: 20px; white-space: pre-wrap; min-height: 200px; }
        .settings { margin: 15px 0; }
        .settings label { margin-right: 15px; }
        .settings input { width: 60px; background: #252526; color: #d4d4d4; border: 1px solid #454545; }
    </style>
</head>
<body>
    <h1>🐍 Tiny Python LLM (6.3M params)</h1>
    <p>Type a Python prompt and the model will complete it.</p>
    
    <textarea id="prompt" placeholder="def factorial(n):">def factorial(n):</textarea>
    
    <div class="settings">
        <label>Max tokens: <input type="number" id="max_tokens" value="50" min="10" max="200"></label>
        <label>Temperature: <input type="number" id="temperature" value="0.8" step="0.1" min="0.1" max="2.0"></label>
        <label>Top-k: <input type="number" id="top_k" value="20" min="1" max="100"></label>
    </div>
    
    <button onclick="generate()">Generate Code</button>
    
    <div id="output"></div>
    
    <script>
        async function generate() {
            const prompt = document.getElementById('prompt').value;
            const max_tokens = document.getElementById('max_tokens').value;
            const temperature = document.getElementById('temperature').value;
            const top_k = document.getElementById('top_k').value;
            const output = document.getElementById('output');
            
            output.textContent = 'Generating...';
            
            const response = await fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({prompt, max_tokens, temperature, top_k})
            });
            
            const data = await response.json();
            output.textContent = data.result;
        }
    </script>
</body>
</html>
"""


# ============ ROUTES ============

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_tokens = int(data.get('max_tokens', 50))
    temperature = float(data.get('temperature', 0.8))
    top_k = int(data.get('top_k', 20))
    
    # Encode prompt
    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
    result = tokenizer.decode(generated[0].tolist())
    
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)