import os
import re

INPUT_DIR = "clean_step1"
OUTPUT_DIR = "clean_step2"

MAX_LINE_LEN = 120
MIN_KEYWORDS = 2

PYTHON_KEYWORDS = ['def ', 'class ', 'import ', 'from ', 'return', 'if ', 'for ', 'while ', 'try:', 'except', 'with ']

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".py")]
print(f"Files to check: {len(files)}")

kept = 0
rejected_long_lines = 0
rejected_no_keywords = 0

for filename in files:
    filepath = os.path.join(INPUT_DIR, filename)
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        lines = content.split('\n')
    
    # Check line lengths
    if any(len(line) > MAX_LINE_LEN for line in lines):
        rejected_long_lines += 1
        continue
    
    # Check for Python keywords
    keyword_count = sum(1 for kw in PYTHON_KEYWORDS if kw in content)
    if keyword_count < MIN_KEYWORDS:
        rejected_no_keywords += 1
        continue
    
    # Save to next clean folder
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    kept += 1

print(f"\nKept: {kept}")
print(f"Rejected (long lines): {rejected_long_lines}")
print(f"Rejected (no keywords): {rejected_no_keywords}")

total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
                 for f in os.listdir(OUTPUT_DIR))
print(f"Clean folder size: {total_size / 1024 / 1024:.1f} MB")