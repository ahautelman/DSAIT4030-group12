import os
import torch
from datasets import load_dataset as hf_load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

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

    transform_list = []
    if num_channels == 3:
        transform_list.append(transforms.Lambda(lambda img: img.convert("RGB")))
        
    transform_list.extend([
        resize,
        transforms.RandomHorizontalFlip(p=random_flip_p),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * num_channels, [0.5] * num_channels)
    ])

    return transforms.Compose(transform_list)


# OPTIMIZATION FIX: A top-level class is perfectly picklable on Windows
class HFTransformWrapper:
    def __init__(self, img_transform, return_classes):
        self.img_transform = img_transform
        self.return_classes = return_classes

    def __call__(self, batch):
        transformed_images = [self.img_transform(img) for img in batch["image"]]
        output = {"images": torch.stack(transformed_images)}
        if self.return_classes and "label" in batch:
            output["class_conditioning"] = torch.tensor(batch["label"])
        return output


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

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if dataset_name == "celeba":
        disk_path = os.path.join(BASE_DIR, "data", "celeba")
        hf_name = "korexyz/celeba-hq-256x256"
        return_classes = False
    elif dataset_name == "imagenet":
        disk_path = os.path.join(BASE_DIR, "data", "imagenet")
        hf_name = "benjamin-paine/imagenet-1k-256x256"
        return_classes = True
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    if not os.path.exists(disk_path):
        print(f"Saving {dataset_name} to disk for faster loading...")
        hf_load_dataset(hf_name).save_to_disk(disk_path)
        print(f"Done! Saved to {disk_path}")

    hf_dataset = load_from_disk(disk_path)[split]

    # Use the picklable class instance instead of a local function
    hf_dataset.set_transform(HFTransformWrapper(img_transform, return_classes))
    return hf_dataset

    
if __name__ == "__main__":
    # Test execution safely protected by the name == main block
    dataset = load_dataset(dataset_name="imagenet", split="train")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
   
    for batch in loader:
        print("Image batch shape:", batch["images"].shape)
        if "class_conditioning" in batch:
            print("Labels:", batch["class_conditioning"])
        break