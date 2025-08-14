import tensorflow as tf
from tensorflow import keras
import numpy as np

class MNISTDataLoader:
    """Data loader for MNIST dataset with preprocessing."""
    
    def __init__(self, normalize=True, reshape_size=(28, 28, 1)):
        self.normalize = normalize
        self.reshape_size = reshape_size
    
    def load_data(self):
        """
        Load and preprocess MNIST dataset.
        
        Returns:
            Tuple of (x_train, y_train), (x_test, y_test)
        """
        # Load raw MNIST data
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Reshape data
        x_train = x_train.reshape(-1, *self.reshape_size)
        x_test = x_test.reshape(-1, *self.reshape_size)
        
        # Normalize pixel values
        if self.normalize:
            x_train = x_train.astype('float32') / 255.0
            x_test = x_test.astype('float32') / 255.0
        
        return (x_train, y_train), (x_test, y_test)
    
    def get_data_info(self, x_train, y_train, x_test, y_test):
        """Print dataset information."""
        print(f"Training samples: {x_train.shape[0]:,}")
        print(f"Test samples: {x_test.shape[0]:,}")
        print(f"Image shape: {x_train.shape[1:]}")
        print(f"Number of classes: {len(np.unique(y_train))}")
