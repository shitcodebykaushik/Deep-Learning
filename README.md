# Tiny Python LLM

A 6.3M parameter Transformer trained from scratch on Python code using PyTorch.

## What's Included

- Custom Byte-level BPE tokenizer (3,000 vocab)
- 6-layer causal Transformer
- Training pipeline with data filtering
- Web interface for code generation

## Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create virtual environment
python -m venv venv

# 3. Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download model checkpoint
# Place `checkpoint_epoch3.pt` and `tokenizer.json` in the 10M/ folder
# (These files are too large for GitHub, see link below)

# 6. Run web interface
python app.py

# 7. Open browser
# Go to http://localhost:5000