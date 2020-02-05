


import os
from PIL import Image
from sklearn.datasets import make_moons


DATA_DIR = "./DATA/mini-dataset/"

#TODO: fix inconsistent return types for data laoders 

def load_raw_data():
    """
    Load apples and bananas images dataset.
    
    Returns
    -------
    images as PIL.Images
    """
    all_files = os.listdir(DATA_DIR)
    all_images = {fname: Image.open(os.path.join(DATA_DIR, fname)) for fname in all_files}
    return all_images


def load_data():
    """
    Load apples and bananas images dataset.
    
    Returns
    -------
    X, y as numpy arrays
    """
    all_images = load_raw_data()
    X = np.array([np.array(img).flatten() for fname, img in all_images.items()])
    y = np.array([fname.startswith('a') for fname, img in all_images.items()])
    return X, y


def load_data_easy():
    import pandas as pd
    file_path = None
    data =  pd.read_csv(file_path)
    X = data.drop('label').values
    y = data['label'].values
    return X, y

def make_data():
    X, y = make_moons(n_samples=500, noise=0.01 )
    return X, y
