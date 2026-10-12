import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    # Embeddings before the update
    v = W_in[center_id].clone()
    u_pos = W_out[pos_id].clone()
    u_neg = W_out[neg_ids].clone()

    # Scores
    pos_score = torch.dot(u_pos, v)
    neg_scores = u_neg @ v

    # Gradients
    pos_coeff = torch.sigmoid(pos_score) - 1
    neg_coeffs = torch.sigmoid(neg_scores)

    grad_v = pos_coeff * u_pos
    grad_v += (neg_coeffs[:, None] * u_neg).sum(dim=0)

    # Copy the original matrices
    new_W_in = W_in.clone()
    new_W_out = W_out.clone()

    # Update center input row
    new_W_in[center_id] -= lr * grad_v

    # Update positive output row
    new_W_out[pos_id] -= lr * pos_coeff * v

    # Update negative rows, including repeated IDs
    for neg_id, coeff in zip(neg_ids, neg_coeffs):
        new_W_out[neg_id] -= lr * coeff * v

    return {
        "W_in": new_W_in,
        "W_out": new_W_out,
    }