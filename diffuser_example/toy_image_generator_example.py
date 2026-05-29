import cv2
import numpy as np
import os
import math

from toy_image_generator import get_toy_image_example_batch

output_dir = "toy_image_examples"
os.makedirs(output_dir, exist_ok=True)

toy_image_batch = get_toy_image_example_batch(
    x_dim=256,
    y_dim=256,
    hue_1s=np.array([0, 60, 120]),
    saturation_1s=np.array([255, 255, 255]),
    value_1s=np.array([255, 255, 255]),
    hue_2s=np.array([90, 150, 70]),
    saturation_2s=np.array([255, 255, 255]),
    value_2s=np.array([255, 255, 255]),
    hue_3s=np.array([0, 0, 0]),
    saturation_3s=np.array([0, 0, 0]),
    value_3s=np.array([0, 0, 0]),
    angles=np.array([0, 45, 90]),
    line_widths=np.array([5, 10, 15])
)

for i in range(toy_image_batch.shape[0]):

    img = toy_image_batch[i]

    # Convert [-1, 1] to [0, 255]
    img = (img + 1.0) / 2.0
    img = (img * 255.0).clamp(0, 255).byte()

    # Convert CxHxW to HxWxC
    img = img.permute(1, 2, 0).cpu().numpy()
    cv2.imwrite(os.path.join(output_dir, f"{i}.png"), img)

