import torch 
import time 

device = "cuda" 

# Use an underscore for readability, NOT a comma
N = 80_00
print(f"Running compute-heacy test with the {N}x{N} matrix size...")

A = torch.rand(N, N, device=device)
B = torch.rand(N, N, device=device)

torch.cuda.synchronize()
start = time.time()
C = torch.matmul(A, B)
torch.cuda.synchronize()
end = time.time()
print("Time taken (GPU compute):", end - start, "seconds")
