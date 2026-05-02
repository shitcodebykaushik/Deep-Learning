import torch
import time

device = "cuda"

x = torch.rand(20000, 2000, device=device)

torch.cuda.synchronize()
start = time.time()

for _ in range(50):
    x = x * 1.0001  # memory-heavy operation

torch.cuda.synchronize()
end = time.time()

print("Memory-heavy time:", end - start)