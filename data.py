import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST, EMNIST
from torchvision.transforms import ToTensor, Normalize, Compose


transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
fdataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)
dataloader = DataLoader(fdataset, batch_size=64, shuffle=True, num_workers=4)

images, labels = next(iter(dataloader))
print(f"{images.shape=} {labels.shape=}")