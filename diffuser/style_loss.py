## ALP-ADDITION: For creating the style loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


def gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """
    features: [B, C, H, W]
    returns: [B, C, C]
    """
    b, c, h, w = features.shape
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))
    return gram / (c * h * w)


class VGGGramStyleLoss(nn.Module):
    """
    Computes Gatys-style Gram matrix loss against one fixed style image.
    """

    def __init__(
        self,
        style_image_path: str,
        device,
        image_size: int = 64,
        layer_ids=(0, 5, 10, 19, 28), # conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    ):
        super().__init__()

        self.device = device
        self.layer_ids = set(layer_ids)

        weights = models.VGG19_Weights.DEFAULT
        vgg = models.vgg19(weights=weights).features.eval().to(device)

        for i, layer in enumerate(vgg):
            if isinstance(layer, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        style_image = self._load_style_image(style_image_path, image_size).to(device)

        with torch.no_grad():
            self.style_grams = [
                gram.detach()
                for gram in self._extract_grams(style_image)
            ]

    def _load_style_image(self, path: str, image_size: int) -> torch.Tensor:
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        image = Image.open(path).convert("RGB")
        image = transform(image).unsqueeze(0)
        return image

    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expects x in [0, 1].
        """
        x = x.clamp(0.0, 1.0)
        return (x - self.imagenet_mean) / self.imagenet_std

    def _extract_grams(self, x: torch.Tensor):
        x = self._normalize_for_vgg(x)

        grams = []
        for layer_idx, layer in enumerate(self.vgg):
            x = layer(x)

            if layer_idx in self.layer_ids:
                grams.append(gram_matrix(x))

            if layer_idx >= max(self.layer_ids):
                break

        return grams

    def forward(self, generated_images: torch.Tensor) -> torch.Tensor:
        """
        generated_images: [B, 3, H, W], expected approximately in [0, 1]
        """
        generated_grams = self._extract_grams(generated_images)

        total_loss = 0.0

        for generated_gram, style_gram in zip(generated_grams, self.style_grams):
            target = style_gram.expand_as(generated_gram)
            total_loss = total_loss + F.mse_loss(generated_gram, target)

        return total_loss