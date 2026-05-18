# define imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 0.001
batch_size = 32
epochs = 30

# Define transformations for the dataset, what transformations does it need?

transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# loading training data from open datasets, root=where, split=train or test dataset 
train_dataset = datasets.OxfordIIITPet(
    root="data",
    split="trainval",
    target_types="category",
    download=True,
    transform=transform
)

# loading the test data from datasets
test_dataset = datasets.OxfordIIITPet(
    root="data",
    split="test",
    target_types="category",
    download=True,
    transform=transform
)

# creating datalloaders which feed data into the model during training
# batch_size=32 means we will fed 32 images at a time, shuffle=True means data is shuffled during training to improve generalization, shuffle=False for test data to maintain order
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# define neural network architecture- defines how data flows through the model, how many layers exist, what they do and how the model learns patterns
# 37 pet categories, so output layer has 37 neurons, input images are resized to 224x224, so input layer has 224*224*3 neurons (3 for RGB channels)

class SimpleNN(nn.Module):
    # parent model required so pytorch can track layers, store parameters and calculate gradients during training. We call super() to initialize the parent class.
    def __init__(self):
        super().__init__()

        # Builds layers in order.
        self.network = nn.Sequential(

            # Convolution Block 1, input channels=3 (red/green/blue), output channels=32 (filters/features), kernel size=3 (3x3 filter that slides along the image), padding=1 to preserve spatial size
            # ReLU activation function introduces non-linearity
            # MaxPool2d with kernel size 2 reduces spatial dimensions by half (downsampling)
            #increseas to learm higher and higher level pattens
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Convolution Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            #converts into 1d vector
            nn.Flatten(),

            # compresses learned features into 512 neurons
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),

            nn.Dropout(0.5),
            # 37 class scores
            nn.Linear(512, 37)
        )
    # defines how data flows through the network
    def forward(self, x):
        return self.network(x)


# Initialize the model, loss function, and optimizer
#loss function measures how wrong the model is
#optimiser updates the model to improve it
model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train(model, train_loader, criterion, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
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

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")


# Evaluation loop
def evaluate(model, test_loader):
    model.eval()

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

    print(f"Accuracy: {100 * correct / total:.2f}%")


# Train and evaluate the model
train(model, train_loader, criterion, optimizer, epochs)

evaluate(model, test_loader)