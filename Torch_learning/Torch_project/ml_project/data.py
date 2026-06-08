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
