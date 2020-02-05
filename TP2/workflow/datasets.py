


import os
from PIL import Image

DATA_DIR = "./DATA/mini-dataset/"


def load_data():
    """
    Load apples and bananas images dataset.
    
    Returns
    -------
    images as PIL.Images
    """
    all_files = os.listdir(DATA_DIR)
    all_images = [Image.open(os.path.join(DATA_DIR, fname)) for fname in all_files]
    return all_images

