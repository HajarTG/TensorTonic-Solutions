import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:

    h = np.dot(x, W1) + b1
    #add ReLU
    h = np.maximum(0, h)
    FFN = np.dot(h, W2) + b2
    return FFN 
