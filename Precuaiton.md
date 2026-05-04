# Always move data to GPU (Explicitly)
- Pytorch does not do this automatcially 
device = "cuda"
x = x.to(device)
model = model.to(device)

- Keep everything on SAME device 
x = x.to("cuda")
y= y.to("cpu")
z = x+y # this is error 
- Avoid CPU <=> GPU transfer 
- Use batch processing as the gpu loves parralle work 
- GPU is bad at loading small data so always load big data 
- Clear gpu memeory when needed 


