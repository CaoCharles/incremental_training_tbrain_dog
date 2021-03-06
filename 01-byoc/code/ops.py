import torch.optim as optim

# torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
# lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
def AdamW(parameters, lr=0.001, betas=(0.9, 0.999), weight_decay=0.01):
    """
    Args:
        parameters (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) – learning rate
        betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square
        weight_decay (float, optional) – weight decay (L2 penalty)
    """
    return optim.AdamW(parameters, lr=lr, betas=betas, weight_decay=weight_decay)

def Adam(parameters, lr=0.0001, betas=(0.9, 0.999), weight_decay=0):
    """
    Args:
        parameters (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional) – learning rate
        betas (Tuple[float, float], optional) – coefficients used for computing running averages of gradient and its square
        weight_decay (float, optional) – weight decay (L2 penalty)
    """
    return optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)

def StepLR(optimizer, step_size=30, gamma=0.1):
    """
    Args:
        optimizer (Optimizer) – Wrapped optimizer.
        step_size (int) – Period of learning rate decay.
        gamma (float) – Multiplicative factor of learning rate decay. Default: 0.1.
    """
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

