# train.py

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

# Reproducibility

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters

learning_rate = 0.0001
batch_size = 32
epochs = 30


# Transforms

train_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = v2.Compose([
    v2.Resize((224, 224)),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Datasets

train_dataset = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="category",
    download=True,
    transform=train_transform
)

test_dataset = datasets.OxfordIIITPet(
    root="data",
    split="test",
    target_types="category",
    download=True,
    transform=test_transform
)


# DataLoaders

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# Model

class SimpleNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 37)
        )

    def forward(self, x):
        return self.network(x)


# Initialize

model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# Training Function

def train(model, train_loader, criterion, optimizer, epochs):

    best_test_accuracy = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Loss: {avg_loss:.4f}")



# Run Training

train(
    model,
    train_loader,
    criterion,
    optimizer,
    epochs
)
