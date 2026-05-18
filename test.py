#test
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Transform

test_transform = v2.Compose([
    v2.Resize((224, 224)),

    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),

    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#Test Dataset

test_dataset = datasets.OxfordIIITPet(
    root="data",
    split="test",
    target_types="category",
    download=True,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

train_dataset = datasets.OxfordIIITPet(
    root="data",    
    split="trainval",
    target_types="category",
    download=True,
    transform=test_transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False
)
#Model Definition

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

#Load Model

model = SimpleNN().to(device)

model.load_state_dict(
    torch.load("model.pth", map_location=device)
)

model.eval()


#Evaluation

def evaluate(test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


print(f"Test Accuracy: {evaluate(test_loader):.2f}%")
print(f"Train Accuracy: {evaluate(train_loader):.2f}%")

