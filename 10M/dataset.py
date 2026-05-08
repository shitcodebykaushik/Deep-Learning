from datasets import load_dataset

# The Stack - Python only, small slice
dataset = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)

# Save first 10,000 examples to files
for i, example in enumerate(dataset):
    if i >= 10000:
        break
    with open(f"python_code_{i}.py", "w") as f:
        f.write(example["content"])