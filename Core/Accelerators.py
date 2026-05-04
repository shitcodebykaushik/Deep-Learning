# Within the pyTorch repo we define an  Accelerator  as a torch.device that  is being  used alongside CPU to speed ip computation . 
# These device uses an asynchronous executionn scheme usign the torch.scheme and torch.event  as theri main way to syncheronization .
# Stream = A queue of task executed in FIFO order on a specific device 
# Event = A way to record when a specific task is completed on a device
