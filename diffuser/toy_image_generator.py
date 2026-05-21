import cv2
import numpy as np
import torch
import math

def get_toy_image_example_batch(x_dim, y_dim, hue_1s, saturation_1s, value_1s, hue_2s, saturation_2s, value_2s, hue_3s, saturation_3s, value_3s, angles, line_widths):
    """
    This function generates a batch of toy images, where each image consists of 2 coloured regions with a coloured line seperating them.
    
    :param x_dim: The width of each image in pixels.
    :param y_dim: The height of each image in pixels.

    :param hue_1s: A array of hues for the first coloured region.
    :param saturation_1s: A array of saturations for the first coloured region.
    :param value_1s: A array of values for the first coloured region.

    :param hue_2s: A array of hues for the second coloured region.
    :param saturation_2s: A array of saturations for the second coloured region.
    :param value_2s: A array of values for the second coloured region.

    :param hue_3s: A array of hues for the coloured line.
    :param saturation_3s: A array of saturations for the coloured line.
    :param value_3s: A array of values for the coloured line.

    :param angles: A array of angles (in degrees) for the orientation of the line.
    :param line_widths: A array of widths for the coloured line.

    :return: A batch of toy images as a PyTorch tensor of shape (batch_size, 3, y_dim, x_dim).
    """ 

    # Assert that all input arrays have the same batch size.
    #-------------------------------------------------
    assert hue_1s.shape[0] == saturation_1s.shape[0]
    assert saturation_1s.shape[0] == value_1s.shape[0]
    assert value_1s.shape[0] == hue_2s.shape[0]
    #-------------------------------------------------
    assert hue_2s.shape[0] == saturation_2s.shape[0]
    assert saturation_2s.shape[0] == value_2s.shape[0]
    assert value_2s.shape[0] == hue_3s.shape[0]
    #-------------------------------------------------
    assert hue_3s.shape[0] == saturation_3s.shape[0]
    assert saturation_3s.shape[0] == value_3s.shape[0]
    assert value_3s.shape[0] == angles.shape[0]
    #-------------------------------------------------
    assert angles.shape[0]== line_widths.shape[0]
    assert line_widths.shape[0]== hue_1s.shape[0]
    #-------------------------------------------------

    # Define image as HxWx3.
    image_resolution = (y_dim, x_dim, 3)

    # Define the integer center coordinate, which the line will always pass through.
    center_coordinate = (y_dim//2, x_dim//2)
    cx, cy = center_coordinate

    # The largest line is the diagonal, so compute the diagonal based on the width and height of the image.
    largest_line = math.sqrt(y_dim**2 + x_dim**2)

    image_list = []
    
    # Repeat for each image in the requested image batch size:
    for batch_n in range(angles.shape[0]):

        # Define starting parameters.
        #-------------------------------------------------------
        image = np.ones(image_resolution, dtype=np.uint8) * 255
        #-------------------------------------------------------
        angle = np.deg2rad(angles[batch_n])
        #-------------------------------------------------------
        hue_1 = hue_1s[batch_n]
        hue_2 = hue_2s[batch_n]
        hue_3 = hue_3s[batch_n]
        #-------------------------------------------------------
        saturation_1 = saturation_1s[batch_n]
        saturation_2 = saturation_2s[batch_n]
        saturation_3 = saturation_3s[batch_n]
        #-------------------------------------------------------
        value_1 = value_1s[batch_n]
        value_2 = value_2s[batch_n]
        value_3 = value_3s[batch_n]
        #-------------------------------------------------------
        line_width = line_widths[batch_n]
        #-------------------------------------------------------

        # Based on the angle, define the direction vector (dx, dy) for the line.
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Compute the endpoints of the line based on the center coordinate and the direction vector, scaled by the largest line length.
        x1 = int(cx - largest_line * dx)
        y1 = int(cy - largest_line * dy)
        x2 = int(cx + largest_line * dx)
        y2 = int(cy + largest_line * dy)

        # Create a grid of pixel coordinates and determine which side of the line each pixel is on using the cross product.
        xs, ys = np.meshgrid(np.arange(image_resolution[1]), np.arange(image_resolution[0]))
        side = (xs - x1) * (y2 - y1) - (ys - y1) * (x2 - x1)

        # Convert HSV colors to BGR for OpenCV drawing.
        #-------------------------------------------------------------
        color1_hsv = np.uint8([[[hue_1, saturation_1, value_1]]])
        color1_bgr = cv2.cvtColor(color1_hsv, cv2.COLOR_HSV2BGR)[0][0]
        #-------------------------------------------------------------
        color2_hsv = np.uint8([[[hue_2, saturation_2, value_2]]])
        color2_bgr = cv2.cvtColor(color2_hsv, cv2.COLOR_HSV2BGR)[0][0]
        #-------------------------------------------------------------
        color3_hsv = np.uint8([[[hue_3, saturation_3, value_3]]])
        color3_bgr = cv2.cvtColor(color3_hsv, cv2.COLOR_HSV2BGR)[0][0]
        #-------------------------------------------------------------

        # Color the image based on which side of the line each pixel falls.
        image[side > 0] = color1_bgr
        image[side < 0] = color2_bgr
        
        # Draw the line with the specified color and line width.
        cv2.line(image, (x1, y1), (x2, y2), (int(color3_bgr[0]), int(color3_bgr[1]), int(color3_bgr[2])), line_width)

        # Convert the image to a PyTorch tensor and normalize to [-1, 1].
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        tensor = torch.sub(torch.mul(tensor, 2.0), 1.0)
        image_list.append(tensor)

    # Stack the list of tensors into a single batch tensor of shape (batch_size, 3, y_dim, x_dim) and return.
    return torch.stack(image_list, dim=0)
