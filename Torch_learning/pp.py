import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ray import train, tune
from ray.air import session
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_data():
    x = torch.randn(1000, 784)
    y = torch.randint(0, 10, (1000,))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)
    return loader


def trainable(config):

    model = MLP(hidden_size=config["hidden_size"], dropout=config["dropout"])

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    loss_fn = nn.CrossEntropyLoss()

    loader = get_data()

    # TRAIN LOOP
    for epoch in range(3):
        total_loss = 0

        for x, y in loader:
            optimizer.zero_grad()

            preds = model(x)
            loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 🔥 WALIDACJA (tutaj uproszczona)
        accuracy = 1.0 / (1.0 + total_loss)  # pseudo accuracy

        # 📡 wysyłasz wynik do Ray Tune
        train.report({"accuracy": accuracy})


results = tune.run(
    trainable,
    config={
        "lr": tune.uniform(1e-5, 1e-2),
        "dropout": tune.uniform(0.1, 0.5),
        "hidden_size": tune.choice([128, 256, 512]),
    },
    num_samples=5,
)

best_trial = results.get_best_trial(metric="accuracy", mode="max", scope="last")

print("\n=== BEST TRIAL ===")
assert best_trial is not None

print(
    "accuracy    :", best_trial.last_result["accuracy"]
)  # "last_result" is not a known attribute of "None"
print("lr          :", best_trial.config["lr"])
print("dropout     :", best_trial.config["dropout"])
print("hidden_size :", best_trial.config["hidden_size"])

for i, trial in enumerate(results.trials):
    print(f"\nMODEL {i + 1}")

    print("accuracy    :", trial.last_result["accuracy"])
    print("lr          :", trial.config["lr"])
    print("dropout     :", trial.config["dropout"])
    print("hidden_size :", trial.config["hidden_size"])
