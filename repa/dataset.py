import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_celeba_dataloader(data_dir: str, batch_size: int, num_workers: int = 4) -> DataLoader:
    """
    Creates a DataLoader for the CelebA dataset.
    Note: torchvision.datasets.ImageFolder expects subdirectories representing classes.
    If 'img_align_celeba' directly contains images, place them in a subfolder (e.g., img_align_celeba/images/).
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     # ImageNet means and standard deviations
    ])

    # Ensure the path exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset path {data_dir} not found.")

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader