import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

IMG_SIZE = 192
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CSV_PATH = "labels.csv"
IMG_DIR = "images"

class FallDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.data = self.data.dropna(subset=["label"])
        self.data = self.data[self.data["label"].isin(["standing", "sitting", "lying", "bending", "crawling"])]
        self.image_dir = image_dir
        self.transform = transform
        self.label_map = {
            'standing': 0,
            'sitting': 1,
            'lying': 2,
            'bending': 3,
            'crawling': 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_path'])
        label = self.label_map[row['label']]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

print("Loading data and splitting 80/20...")
full_dataset = FallDataset(CSV_PATH, IMG_DIR, transform)
train_len = int(0.8 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_set, val_set = random_split(full_dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print("Preparing MobileNetV2 model...")
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_loss_list, val_loss_list = [], []
train_acc_list, val_acc_list = [], []
best_val_acc = 0.0

print("Start training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    train_acc = correct / len(train_loader.dataset)
    train_loss_list.append(total_loss)
    train_acc_list.append(train_acc)

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}, Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "pose_classifier.pt")

print("Exporting ONNX model...")
model.eval()
dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
torch.onnx.export(
    model, dummy_input, "pose_classifier.onnx",
    input_names=["input"], output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

plt.figure()
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")

plt.figure()
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy Curve")
plt.legend()
plt.grid(True)
plt.savefig("acc_curve.png")

print("loss_curve.png and acc_curve.png have been saved.")
print("Training complete! Model and charts have been exported.")
