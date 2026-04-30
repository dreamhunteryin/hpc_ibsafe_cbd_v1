import torch
import torch.nn.functional as F


def _sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    prob = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return alpha_t * ce_loss * ((1.0 - p_t) ** gamma)


def triton_sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    return _sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma)


def triton_sigmoid_focal_loss_reduce(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    return _sigmoid_focal_loss(inputs, targets, alpha=alpha, gamma=gamma).sum()
