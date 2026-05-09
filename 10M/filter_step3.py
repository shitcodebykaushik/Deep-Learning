import os

INPUT_DIR = "clean_step2"
OUTPUT_DIR = "clean_step3"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".py")]
print(f"Files to clean: {len(files)}")

for filename in files:
    filepath = os.path.join(INPUT_DIR, filename)
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Replace tabs with 4 spaces
    content = content.replace('\t', '    ')
    
    # Split into lines
    lines = content.split('\n')
    
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in lines]
    
    # Remove leading blank lines
    while lines and lines[0] == '':
        lines.pop(0)
    
    # Remove trailing blank lines
    while lines and lines[-1] == '':
        lines.pop()
    
    # Collapse 3+ consecutive blank lines to 2
    cleaned_lines = []
    blank_count = 0
    
    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)
    
    # Rebuild content
    content = '\n'.join(cleaned_lines) + '\n'
    
    # Save
    out_path = os.path.join(OUTPUT_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

total_size = sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) 
                 for f in os.listdir(OUTPUT_DIR))
print(f"Done. Clean folder size: {total_size / 1024 / 1024:.1f} MB")