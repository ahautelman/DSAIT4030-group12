from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader


def image_transforms(num_channels=3, img_size=256, random_resize=True, interpolation="bilinear", random_flip_p=0.5, split="train"):

    interpolation_dict = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC
    }

    if random_resize and split == "train":
        resize = transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=interpolation_dict[interpolation])
    else:
        resize = transforms.Resize((img_size, img_size))

    if split != "train":
        random_flip_p = 0.0

    image2tensor = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB") if num_channels == 3 else img),
        resize,
        transforms.RandomHorizontalFlip(p=random_flip_p),
        transforms.ToTensor(),                  # [0, 1]
        transforms.Normalize(
            [0.5] * num_channels,
            [0.5] * num_channels
        ),                                       # [-1, 1]
    ])

    return image2tensor


class HFImageDataset(Dataset):
    """
    Wraps a HuggingFace dataset for use with PyTorch DataLoader.

    Args:
        hf_dataset:     A HuggingFace Dataset object (already downloaded, not streaming)
        transform:      torchvision transform pipeline
        return_classes: Whether to return class labels (only for ImageNet)
    """

    def __init__(self, hf_dataset, transform=None, return_classes=False):
        self.dataset = hf_dataset
        self.transform = transform
        self.return_classes = return_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["image"]

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_classes:
            label = sample["label"]
            return {"images": img, "class_conditioning": label}
        else:
            return {"images": img}


def load_dataset(dataset_name: str, num_channels=3, img_size=256, random_resize=True,
                 interpolation="bilinear", random_flip_p=0.5, split="train"):

    img_transform = image_transforms(
        num_channels=num_channels,
        img_size=img_size,
        random_resize=random_resize,
        interpolation=interpolation,
        random_flip_p=random_flip_p,
        split=split,
    )

    if dataset_name == "imagenet":
        hf_dataset = hf_load_dataset("benjamin-paine/imagenet-1k-256x256", split=split)
        return HFImageDataset(hf_dataset, transform=img_transform, return_classes=True)
    elif dataset_name == "celeba":
        hf_dataset = hf_load_dataset("korexyz/celeba-hq-256x256", split=split)
        return HFImageDataset(hf_dataset, transform=img_transform, return_classes=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")


if __name__ == "__main__":
    dataset = load_dataset(dataset_name="imagenet", split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for batch in loader:
        print("Image batch shape:", batch["images"].shape)
        print("Labels:", batch["class_conditioning"])
        break