import cv2
import numpy as np
import torch
import math

def get_toy_image_example_batch(x_dim, y_dim, hue_1s, saturation_1s, value_1s, hue_2s, saturation_2s, value_2s, hue_3s, saturation_3s, value_3s, angles, line_widths):
    """
    This function generates a batch of toy images, i.e. an image with two coloured regions and a line through the center separating these regions.
    
    input:
    x_dim, y_dim: WxH of images
    hue_1s, saturation_1s, value_1s: HSV colours for region 1
    hue_2s, saturation_2s, value_2s: HSV colours for region 2
    hue_3s, saturation_3s, value_3s: HSV colours for line
    angles: Orientation angles of the line in degrees
    line_widths: Thicknesses of the lines

    return: Normalized  batch of toy images
    """

    # All parameters should correlate to the same batch size
    assert hue_1s.shape[0] == saturation_1s.shape[0]
    assert saturation_1s.shape[0] == value_1s.shape[0]
    assert value_1s.shape[0] == hue_2s.shape[0]
    assert hue_2s.shape[0] == saturation_2s.shape[0]
    assert saturation_2s.shape[0] == value_2s.shape[0]
    assert value_2s.shape[0] == hue_3s.shape[0]
    assert hue_3s.shape[0] == saturation_3s.shape[0]
    assert saturation_3s.shape[0] == value_3s.shape[0]
    assert value_3s.shape[0] == angles.shape[0]
    assert angles.shape[0]== line_widths.shape[0]
    assert line_widths.shape[0]== hue_1s.shape[0]

    # General line parameters valid for all images
    center_coordinate = (y_dim//2, x_dim//2)
    cx, cy = center_coordinate
    largest_line = math.sqrt(y_dim**2 + x_dim**2)

    image_list = []
    
    for batch_n in range(angles.shape[0]):

        # Convert HSV parameters to BGR colours for opencv
        color1_hsv = np.uint8([[[hue_1s[batch_n], saturation_1s[batch_n], value_1s[batch_n]]]])
        color1_bgr = cv2.cvtColor(color1_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color2_hsv = np.uint8([[[hue_2s[batch_n], saturation_2s[batch_n], value_2s[batch_n]]]])
        color2_bgr = cv2.cvtColor(color2_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color3_hsv = np.uint8([[[hue_3s[batch_n], saturation_3s[batch_n], value_3s[batch_n]]]])
        color3_bgr = cv2.cvtColor(color3_hsv, cv2.COLOR_HSV2BGR)[0][0]

        # Define the direction vector and the endpoints of the line
        dx = math.cos(np.deg2rad(angles[batch_n]))
        dy = math.sin(np.deg2rad(angles[batch_n]))
        x1 = int(cx - largest_line * dx)
        y1 = int(cy - largest_line * dy)
        x2 = int(cx + largest_line * dx)
        y2 = int(cy + largest_line * dy)

        # Color the coloured regions relative to the line
        image = np.ones((y_dim, x_dim, 3), dtype=np.uint8) * 255
        xs, ys = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        side = (xs - x1) * (y2 - y1) - (ys - y1) * (x2 - x1)
        image[side > 0] = color1_bgr
        image[side < 0] = color2_bgr
        
        # Draw the line over the coloured regions
        cv2.line(image, (x1, y1), (x2, y2), (int(color3_bgr[0]), int(color3_bgr[1]), int(color3_bgr[2])), line_widths[batch_n])

        # Convert to a PyTorch tensor and normalize
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = torch.sub(torch.mul(tensor, 2.0), 1.0)
        image_list.append(tensor)
        
    return torch.stack(image_list, dim=0)
