import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Add project root to path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mlp import MLP
from data.dataset_loader import DatasetLoader

# === Configuration ===
DATA_PATHS = [
    "data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "data/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "data/MachineLearningCVE/Wednesday-workingHours.pcap_ISCX.csv",
]
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
VALID_SPLIT = 0.2

# === Load datasets ===
loader = DatasetLoader()
Xs, ys = [], []

print("[train.py] Loading and combining datasets...")
for path in DATA_PATHS:
    try:
        X, y = loader.load(path)
        if y is None or np.all(y == 0):
            print(f"[train.py] Skipping unlabeled dataset: {path}")
            continue
        Xs.append(X)
        ys.append(y)
    except Exception as e:
        print(f"[train.py] Error loading {path}: {e}")

if not Xs or not ys:
    raise RuntimeError("[train.py] No valid datasets loaded.")

X = np.vstack(Xs)
y = np.concatenate(ys)
print(f"[DEBUG] Combined shape: X={X.shape}, y={y.shape}")

# === Encode labels
if y.dtype == object or not np.issubdtype(y.dtype, np.integer):
    print("[train.py] Encoding string labels to integers...")
    y = LabelEncoder().fit_transform(y)

# === Prepare PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)

# === Train/validation split
val_size = int(len(dataset) * VALID_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Initialize model
input_dim = X.shape[1]
output_dim = len(np.unique(y))
model = MLP(input_dim=input_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === Training loop with validation
train_losses = []
val_accuracies = []

print(f"[train.py] Training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    val_accuracy = correct / total
    val_accuracies.append(val_accuracy)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

print("[train.py] Training complete.")

# === Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === (Optional) Save model
# torch.save(model.state_dict(), "saved_model.pth")




