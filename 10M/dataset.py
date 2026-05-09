import os
from datasets import load_dataset

# Configuration
MY_TOKEN = "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" 
OUTPUT_DIR = "raw_data"
LIMIT = 10000

# Create directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Connecting to Hugging Face...")
dataset = load_dataset(
    "bigcode/the-stack", 
    data_dir="data/python", 
    split="train", 
    streaming=True,
    token=MY_TOKEN
)

print(f"Starting download of {LIMIT} Python files...")

for i, example in enumerate(dataset):
    if i >= LIMIT:
        break
    
    # Using utf-8 encoding is essential for Windows
    file_path = os.path.join(OUTPUT_DIR, f"python_code_{i}.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(example["content"])
    
    # Print progress every 500 files
    if i % 500 == 0 and i > 0:
        print(f"Downloaded {i} files...")

print(f"Done! All files are saved in the '{OUTPUT_DIR}' folder.")