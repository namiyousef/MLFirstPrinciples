import torch

def polynomial_kernel(xi, xt, d):
    return (xi @ xt.T) ** d


def gaussian_kernel(xi, xt, c):
    # TODO make sure this works for batch size
    xii = xi @ xi.T
    xii = torch.diag(xii).reshape(-1,1) if xii.size() else xii
    xtt = xt @ xt.T
    xtt = torch.diag(xtt) if xtt.size() else xtt
    xit = xi @ xt.T
    return torch.exp(c*(2*xit - xii - xtt))