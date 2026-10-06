import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    frequencies = counts / counts.sum()
    inverse = torch.sqrt(t / frequencies)
    return torch.where(inverse <= 1, inverse, 1)
    
    