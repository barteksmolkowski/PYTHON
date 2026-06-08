import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from model import MLP

from data import get_loader  # "get_loader" is unknown import symbol

"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loader(batch_size):
    return DataLoader(
        datasets.MNIST(
            root="./data", train=True, download=True, transform=transforms.ToTensor()
        ),
        batch_size=batch_size,
        shuffle=True,
    )

plik data:

"""


# LOAD CONFIG
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg["device"])

# DATA
loader = get_loader(cfg["batch_size"])

# MODEL
model = MLP(cfg["input_size"], cfg["hidden_size"], cfg["num_classes"]).to(device)

# LOSS + OPT
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

# TRAIN
for epoch in range(cfg["epochs"]):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("epoch", epoch, "loss:", total_loss / len(loader))
