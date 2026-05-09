import os

INPUT_DIR = "clean_step3"
OUTPUT_FILE = "python_train.txt"
SEPARATOR = "\n<|endoftext|>\n"

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".py")]
files.sort()

print(f"Merging {len(files)} files...")

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for i, filename in enumerate(files):
        filepath = os.path.join(INPUT_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        out.write(content)
        out.write(SEPARATOR)
        
        if (i + 1) % 1000 == 0:
            print(f"  Merged {i + 1} files...")

final_size = os.path.getsize(OUTPUT_FILE)
print(f"\nDone! Output: {OUTPUT_FILE}")
print(f"Size: {final_size / 1024 / 1024:.1f} MB")
print(f"Files merged: {len(files)}")