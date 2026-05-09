from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# Initialize
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

# Trainer config
trainer = BpeTrainer(
    vocab_size=3000,
    special_tokens=["<|endoftext|>"],
    min_frequency=2,
    show_progress=True
)

# Train on your merged file
print("Training tokenizer... This may take a few minutes.")
tokenizer.train(files=["python_train.txt"], trainer=trainer)

# Save
tokenizer.save("tokenizer.json")
print("Saved: tokenizer.json")

# Test it
test_code = "def hello_world():\n    print('Hello!')"
encoded = tokenizer.encode(test_code)
print(f"\nTest: {test_code}")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")
print(f"Length: {len(encoded.ids)} tokens")