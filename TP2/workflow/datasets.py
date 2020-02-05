


import os
from PIL import Image
from sklearn.datasets import make_moons


DATA_DIR = "./DATA/mini-dataset/"

#TODO: fix inconsistent return types for data laoders 

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



def make_data():
    X, y = make_moons(n_samples=500, noise=0.01 )
    return X, y
