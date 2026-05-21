import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_celeba_dataloader(data_dir: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])     # normalize for VAE
    ])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path {data_dir} not found.")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,          
        drop_last=True,
        persistent_workers=True
    )
    return dataloader