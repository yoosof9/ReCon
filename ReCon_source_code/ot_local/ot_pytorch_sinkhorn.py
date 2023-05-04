import torch

def sinkhorn(M, r=None, c=None, gamma=10.0, eps=1.0e-6, maxiters=2, logspace=False):
    """
    PyTorch function for entropy regularized optimal transport. Assumes batched inputs as follows:
        M:  (B,H,W) tensor
        r:  (B,H) tensor, (1,H) tensor or None for constant uniform vector 1/H
        c:  (B,W) tensor, (1,W) tensor or None for constant uniform vector 1/W

    You can back propagate through this function in O(TBWH) time where T is the number of iterations taken to converge.
    """

    B, H, W = M.shape
    assert r is None or r.shape == (B, H) or r.shape == (1, H)
    assert c is None or c.shape == (B, W) or c.shape == (1, W)
    assert not logspace or torch.all(M > 0.0)

    r = 1.0 / H if r is None else r.unsqueeze(dim=2)
    c = 1.0 / W if c is None else c.unsqueeze(dim=1)

    if logspace:
        P = torch.pow(M, gamma)
    else:
        P = torch.exp(-1.0 * gamma * (M - torch.amin(M, 2, keepdim=True)))

    for i in range(maxiters):
        alpha = torch.sum(P, 2)
        # Perform division first for numerical stability
        P = P / alpha.view(B, H, 1) * r

        beta = torch.sum(P, 1)
        if torch.max(torch.abs(beta - c)) <= eps:
            break
        P = P / beta.view(B, 1, W) * c

    return P

