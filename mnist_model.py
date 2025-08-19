# train.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

# Chọn device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================
# CNN mạnh hơn
# ==========================
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128*5*5, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# ==========================
# Data augmentation + transform
# ==========================
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ==========================
# Model, optimizer
# ==========================
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters())

# ==========================
# Train
# ==========================
num_epochs = 20  # Bạn có thể tăng lên 50–100 nếu GPU mạnh

for epoch in range(1, num_epochs+1):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader,1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        '''if batch_idx % 10 == 0:
            print(f"Epoch {epoch} Batch {batch_idx}, Loss: {loss.item():.4f}")'''

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} finished, Average Loss: {avg_loss:.4f}")

    # Lưu model
    os.makedirs('saved_model', exist_ok=True)
    torch.save(model.state_dict(), f'saved_model/mnist_cnn_epoch{epoch}.pth')
    print(f"Model saved for epoch {epoch}\n")

print("Training finished!")
