import torch 
import time 

x = torch.rand(1000, 1000, device="cuda")

start = time.time()
print("Wrong timing")
y = torch.mm(x, x)
end = time.time()
print(f"Time: {end - start} ")

print("Correct timing")
start = time.time()
y = torch.mm(x, x)
torch.cuda.synchronize()
end = time.time()
print(f"Time: {end - start} ")
