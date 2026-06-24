from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ---------------------------------
# Paths (ALWAYS SAVE INSIDE backend/)
# ---------------------------------
BACKEND_DIR = Path(__file__).resolve().parent          # .../backend
PROJECT_ROOT = BACKEND_DIR.parent                     # .../Agentic_Tumor

DATASET_PATH = PROJECT_ROOT / "dataset" / "Training"

TRAINED_MODELS_DIR = BACKEND_DIR / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Config
# ---------------------------------
BATCH_SIZE = 32
EPOCHS = 5

LR = 0.0003

# ---------------------------------
# Transforms
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------------------------
# Load Dataset
# ---------------------------------
print("📂 Loading dataset from:", DATASET_PATH)

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"❌ Dataset folder not found: {DATASET_PATH}")

train_data = datasets.ImageFolder(
    root=str(DATASET_PATH),
    transform=transform
)

print("✅ Classes:", train_data.classes)
print("✅ Num classes:", len(train_data.classes))

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ---------------------------------
# Model (NO pretrained weights)
# ---------------------------------
print("🧠 Building ResNet18 model...")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

# ---------------------------------
# Training Setup
# ---------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("⚙️ Training on device:", device)

# ---------------------------------
# Training Loop
# ---------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total if total else 0
    print(f"📊 Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.3f} | Accuracy: {acc:.3f}")

# ---------------------------------
# Save Model (INSIDE backend/trained_models)
# ---------------------------------
SAVE_PATH = TRAINED_MODELS_DIR / "tumor_model.pt"
torch.save(model.state_dict(), SAVE_PATH)

print("💾 Model saved at:", SAVE_PATH)
print("✅ Training complete")