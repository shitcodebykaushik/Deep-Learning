import torch 
import time 

device = "cuda"

N = 800

print(f"Running TFLOPS test with  {N}*{N} matrix....")

A = torch.rand(N,N, device=device)
B = torch.rand(N,N, device=device)


for _ in range (3):
    torch.matmul(A,B)

torch.cuda.synchronize()

start = time.time()
C = torch.matmul(A,B)
torch.cuda.synchronize()
end = time.time()

time_taken = end - start

# Calculate TFLOPS
flops = 2 * (N ** 3)
print(flops)
tflops = flops / (time_taken * 1e12)
print(f"Time taken: {time_taken:.6f} seconds")
print(f"TFLOPS: {tflops:.2f}")
print(f"TFLOPS per GPU: {tflops:.2f}")
