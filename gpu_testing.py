import torch

print("----- BASIC CHECK -----")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

print("\n----- DEVICE INFO -----")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device:", device)

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
else:
    print("No GPU detected, using CPU")

print("\n----- COMPUTATION TEST -----")
# Create a tensor and move it to device
x = torch.rand(3, 3).to(device)
y = torch.rand(3, 3).to(device)

z = x + y
print("Tensor device:", z.device)
print("Result:\n", z)