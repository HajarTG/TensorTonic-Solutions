import numpy as np

def softmax(x, axis=-1):
    """Provided: numerically stable softmax."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    nor = (x - mean) / np.sqrt(var + eps)
    return gamma * nor + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    B, T, d = Q.shape
    head_dim = d // num_heads

    Q = np.dot(Q, W_q)
    K = np.dot(K, W_k)
    V = np.dot(V, W_v)

    Q_head = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_head = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_head = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = np.matmul(Q_head, K_head.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    attention = softmax(scores, axis=-1)
    output = np.matmul(attention, V_head)

    final_output = output.transpose(0, 2, 1, 3).reshape(B, T, d)

    MHA = final_output @ W_o

    return MHA
    

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    h = np.dot(x, W1) + b1
    #add ReLU
    h = np.maximum(0, h)
    FFN = np.dot(h, W2) + b2
    return FFN


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    MHA_output = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + MHA_output, gamma1, beta1)
    FFN_output = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + FFN_output, gamma2, beta2)
    return x 