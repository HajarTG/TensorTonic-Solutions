import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    h_0= np.zeros((batch_size,hidden_dim),dtype=np.float64)
    return h_0