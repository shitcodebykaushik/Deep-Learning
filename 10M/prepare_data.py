from tokenizers import Tokenizer
import numpy as np
import os

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Read training text
print("Reading python_train.txt...")
with open("python_train.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize entire file
print("Tokenizing...")
encoded = tokenizer.encode(text)
token_ids = encoded.ids

print(f"Total tokens: {len(token_ids):,}")

# Check separator presence
sep_id = tokenizer.token_to_id("<|endoftext|>")
sep_count = token_ids.count(sep_id)
print(f"Separator tokens: {sep_count} (should be ~5942)")

# Save as numpy array for fast loading
tokens_array = np.array(token_ids, dtype=np.uint16)  # uint16 fits 0-65535, our vocab is 3000
np.save("python_train_tokens.npy", tokens_array)

file_size = os.path.getsize("python_train_tokens.npy")
print(f"Saved: python_train_tokens.npy")
print(f"Size: {file_size / 1024 / 1024:.1f} MB")
print(f"Data type: {tokens_array.dtype}")