import numpy as np
from typing import Tuple
from tensorflow.keras.datasets import mnist

def load_train(padding=((0, 0), (0, 0), (0, 3))) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns training data, X, y. 
    """
    (train_images, train_labels), _ = mnist.load_data()

    return np.pad(train_images, padding), train_labels

def load_test(padding=((0, 0), (0, 0), (3, 0))) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns testing data, X, y. 
    """
    _, (test_images, test_labels) = mnist.load_data()

    return np.pad(test_images, padding), test_labels

def load_example(index=4, paddingL=((0,0), (0, 3)), paddingR=((0,0), (3, 0))):
    """
    Returns one image twice with different paddings
    """
    _, (example_images, example_labels) = mnist.load_data()
    example_image = example_images[index]
    
    return (np.pad(example_image, paddingL), np.pad(example_image, paddingR)), example_labels[index]
