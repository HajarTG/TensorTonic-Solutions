import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)
    
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
    

    
    


    
    