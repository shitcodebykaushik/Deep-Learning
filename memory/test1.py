import torch
import torch.nn.functional as F

# 1. Manual "Memory" Setup (Vocabulary)
vocab = {"my": 0, "dog": 1, "is": 2, "max": 3, "eats": 4, "meat": 5}
inv_vocab = {v: k for k, v in vocab.items()}
d_model = 8  # Meaning vector size

# 2. Embedding Layer (Long-term Memory of word meanings)
embeddings = torch.nn.Embedding(len(vocab), d_model)

def test_attention(sentence):
    # Text -> Tokens -> IDs
    tokens = sentence.lower().split()
    ids = torch.tensor([vocab[token] for token in tokens])
    
    # IDs -> Vectors
    x = embeddings(ids).unsqueeze(0) # Shape: [1, seq_len, d_model]
    
    # 3. Simple Attention Mechanism
    query = x  # In self-attention, Q, K, and V come from the same input
    key = x
    value = x
    
    # Similarity calculation (QK^T)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (d_model**0.5)
    weights = F.softmax(scores, dim=-1)
    
    # Output context
    output = torch.matmul(weights, value)
    
    print(f"\nAnalyzing: '{sentence}'")
    print("-" * 30)
    for i, token in enumerate(tokens):
        print(f"Token '{token}' is paying attention to:")
        for j, score in enumerate(weights[0, i]):
            print(f"  -> {tokens[j]}: {score:.4f}")

# TEST IT
test_attention("my dog is kaushik")