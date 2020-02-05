import numpy as np


class ImageHistogram(object):
    """
    La doc !
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """
        Do nothing
        """
        return self
    
    def transform(self, X):
        """
        ToDo Doc
        """
        X_new = np.array([self._image_histogram(img) for img in X])
        return X_new

    def _image_histogram(self, image):
        count = np.zeros(256)
        for pix_value in image:
            count[pix_value] += 1
        return count

