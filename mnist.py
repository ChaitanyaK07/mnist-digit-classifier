import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── Data Augmentation & Preprocessing ───────────────────────────────────────
# Training: augmentation helps the model generalise better
train_transform = transforms.Compose([
    transforms.RandomRotation(10),          # rotate ±10°  (simulates sloppy handwriting)
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),              # shift up to 10% in x/y
        shear=5                            # slight shear distortion
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # scale to [-1, 1]
])

# Testing: no augmentation — only normalise
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ── Datasets & Loaders ──────────────────────────────────────────────────────
train_dataset = datasets.MNIST(root='./data', train=True,  download=True, transform=train_transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=0)

# ── Model Architecture ───────────────────────────────────────────────────────
class CNN(nn.Module):
    """
    Two-block convolutional network with Batch Normalisation and Dropout.

    Input  : (B, 1, 28, 28)
    Block 1: Conv(1→32) → BN → ReLU → MaxPool  →  (B, 32, 13, 13)
    Block 2: Conv(32→64)→ BN → ReLU → MaxPool  →  (B, 64,  5,  5)
    Head   : Flatten → FC(1600→128) → BN → ReLU → Dropout(0.5) → FC(128→10)
    """
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            # ── Block 1 ──────────────────────────────
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B,32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (B,32,14,14)

            # ── Block 2 ──────────────────────────────
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,14,14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # (B,64, 7, 7)
        )

        # With padding=1 the spatial size halves twice: 28→14→7
        # so the flattened size = 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


model = CNN().to(device)
print(model)

# ── Loss, Optimiser & Scheduler ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Reduce LR by 0.5 if val accuracy stagnates for 2 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

# ── Training Loop ────────────────────────────────────────────────────────────
epochs = 10
train_acc_history = []
test_acc_history  = []

for epoch in range(epochs):
    # ── Train ──────────────────────────────────────
    model.train()
    correct = total = 0
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()          # ← zero BEFORE forward pass
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted  = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_acc_history.append(train_acc)

    # ── Validate ────────────────────────────────────
    model.eval()
    val_correct = val_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    test_acc = 100 * val_correct / val_total
    test_acc_history.append(test_acc)

    scheduler.step(test_acc)   # adjust LR based on val accuracy

    print(f"Epoch [{epoch+1:02d}/{epochs}]  "
          f"Loss: {running_loss/len(train_loader):.4f}  "
          f"Train Acc: {train_acc:.2f}%  "
          f"Test Acc: {test_acc:.2f}%")

# ── Save Model ───────────────────────────────────────────────────────────────
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/mnist_cnn.pth")
print("\nModel saved to model/mnist_cnn.pth")

# ── Plot Accuracy ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_acc_history, marker='o', label='Train Accuracy')
plt.plot(range(1, epochs + 1), test_acc_history,  marker='s', label='Test Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png", dpi=150)
plt.show()
print("Plot saved to accuracy_plot.png")