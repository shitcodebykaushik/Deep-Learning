import torch

# Load full checkpoint
checkpoint = torch.load("10M/checkpoint_epoch3.pt", map_location="cpu")

# Save only model weights
torch.save(checkpoint['model_state_dict'], "10M/model_weights.pt")

print("Saved stripped model: model_weights.pt")


