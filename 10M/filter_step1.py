import os

INPUT_DIR = "raw_data"
OUTPUT_DIR = "clean_step1"

MIN_SIZE = 500        # bytes
MAX_SIZE = 20000      # bytes (20 KB)
MIN_LINES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".py")]
print(f"Total files found: {len(files)}")

kept = 0
rejected_size = 0
rejected_lines = 0

for filename in files:
    filepath = os.path.join(INPUT_DIR, filename)
    
    # Check size
    size = os.path.getsize(filepath)
    if size < MIN_SIZE or size > MAX_SIZE:
        rejected_size += 1
        continue
    
    # Check line count
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except:
        rejected_lines += 1
        continue
    
    if len(lines) < MIN_LINES:
        rejected_lines += 1
        continue
    
    # Copy to clean folder
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    kept += 1

print(f"\nKept: {kept}")
print(f"Rejected (bad size): {rejected_size}")
print(f"Rejected (too few lines): {rejected_lines}")

# Check new folder size
total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
                 for f in os.listdir(OUTPUT_DIR))
print(f"Clean folder size: {total_size / 1024 / 1024:.1f} MB")