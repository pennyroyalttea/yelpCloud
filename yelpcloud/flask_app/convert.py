import torch
model=torch.load('model.pt')
torch.save(model.state_dict(), 'new.pth')
print("done")
