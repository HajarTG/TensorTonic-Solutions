import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int,
                 W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 CBOW cross-entropy loss.
    """
    hidden = W_in[context_ids].mean(dim=0)
    logits = W_out @ hidden
    loss = -F.log_softmax(logits, dim=0)[target_id]
    return loss 