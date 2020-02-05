import numpy as np


def image_histogram(image):
    count = np.zeros(256)
    for pix_value in image:
        count[pix_value] += 1
    return count


def to_histogram(X):
    X_new = np.array([image_histogram(img) for img in X])
    return X_new
