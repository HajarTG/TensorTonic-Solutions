import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    n = len(token_ids)
    result = []
    for center_index in range(n):
        start = max(0, center_index - window)
        stop = min(n, center_index + window + 1)
        for context in range (start, stop):
            if context != center_index:
                result.append([token_ids[center_index], token_ids[context]])
    return torch.tensor(result, dtype=torch.int64).reshape(-1, 2)
        
    
    