from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

train_dataset = datasets.MNIST(
    root="/dataset", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(
    root="/dataset", train=False, transform=transforms.ToTensor(), download=True
)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


### Initalize network ###
class NN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, expansion=1):
        super().__init__()

        self.fc1 = nn.Linear(
            in_features=input_size,
            out_features=hidden_size * expansion,
        )

        self.fc2 = nn.Linear(
            in_features=hidden_size * expansion,
            out_features=num_classes,
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NN(input_size=input_size, num_classes=num_classes).to(device)

### Loss and optimizer ###
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

### Trainer Network ###
data: Tensor
targets: Tensor

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        loss: Tensor = criterion(scores, targets)
        optimizer.zero_grad()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


### Check accuracy on training & test ###
def check_accuracy(loader: DataLoader, model: nn.Module) -> float:
    dataset = cast(MNIST, loader.dataset)

    if dataset.train:
        print("Checking accuracy on training data.")
    else:
        print("Checking accuracy on test data.")

    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            scores = model(inputs)

            _, predictions = scores.max(1)

            num_correct += (predictions == targets).sum().item()
            num_samples += len(predictions)
            running_loss += loss.item()

        print(f"Epoch {epoch}: loss = {running_loss / len(loader)}")

    accuracy = num_correct / num_samples * 100

    print(f"Accuracy: {accuracy:.2f}%")

    model.train()

    return accuracy


if epoch % 1 == 0:
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)
