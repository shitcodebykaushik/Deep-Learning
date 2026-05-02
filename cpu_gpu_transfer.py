import torch
import time

x = torch.rand(10000, 1000)

start = time.time()
for _ in range(100):
    y = x.to("cuda")
torch.cuda.synchronize()
end = time.time()

print("Transfer time:", end - start)