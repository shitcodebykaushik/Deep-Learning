import torch
import time

device = "cuda"

# Use an underscore for readability, NOT a comma
size = 30_000 

# This will now correctly create two 40,000 x 40,000 matrices
print(f"Allocating matrices of size {size}x{size}...")

x = torch.rand(size, size, device=device)
y = torch.rand(size, size, device=device)

torch.cuda.synchronize()
start = time.time()

z = torch.matmul(x, y)

torch.cuda.synchronize()
end = time.time()

print("Time taken (GPU compute):", end - start, "seconds")