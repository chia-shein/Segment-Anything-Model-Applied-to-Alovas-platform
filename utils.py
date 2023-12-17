import os
import cv2
import pyvips
import numpy as np
from typing import Any, Dict, List

MIN_AREA = 5000


def get_points(points: List, level: int):
    """Get points prompt from commandline.
    
    Example:
        [123 456 789 112] -> [[123, 456], [789, 112]]
    """
    input_point = np.array(points)
    num_point = len(input_point)//2
    input_point = np.array(np.split(input_point, num_point))//(4**level)
    input_label = np.array([1]*num_point)
    return input_point, input_label


def load_image(image_path: str, level: int):
    """Load image by using pyvips"""
    image = pyvips.Image.new_from_file(image_path, level=level)
    image = image.colourspace("srgb")
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    for i, mask_data in enumerate(masks):
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask_data * 255)
    return mask_data.astype(np.uint8)


def get_contours(mask):
    if len(mask.shape) > 2:
        mask = np.squeeze(mask, 0)
    mask = mask.astype(np.uint8)
    mask *= 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    effContours = []
    total_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > MIN_AREA:
            effContours.append(c)
            total_area += area
    return effContours, total_area


def contour_to_points(contour):
    pointsNum = len(contour)
    contour = contour.reshape(pointsNum, -1).astype(np.float32)
    points = [point.tolist() for point in contour]
    return points


def generate_json(binary_masks):
    """Generate a JSON file from binary mask tensor.

    Args:
        binary_masks: Binary mask tensor of shape [N, H, W].
        labels: List of labels for each mask.
        image_size: Tuple of (height, width) for the image size.
        image_path: Path to the image file (optional).

    Returns:
        A dictionary representing the JSON file.
    """
    json_dict = {
        "annotation": list(),
        "information": dict()
    }

    # Loop through the masks and add them to the JSON dictionary
    for mask in binary_masks:
        effContours, area = get_contours(mask)

        for effContour in effContours:
            points = contour_to_points(effContour)
            shape_dict = {
                "name": "tissue",
                "type": "polygon",
                "partOfGroup": "HCC",
                "coordinates": points,
            }

            json_dict["annotation"].append(shape_dict)
            json_dict["information"]["Tissue area"] = area

    return json_dict
