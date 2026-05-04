import torch 
import torch.nn as nn

model = nn.Linear(1,1) 

for param in model.parameters():
    print(param)
    
    
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    
    # Forward pass 
    
    y_pred = model(x)
    
    # Compute loss
    loss = loss_fn(y_pred, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
  

    # Update weights
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        
test = torch.tensor([[5.0]])
prediction = model(test)

print("Prediction for 5:", prediction.item())