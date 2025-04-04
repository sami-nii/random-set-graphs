import torch

def interval_softmax(a_L, a_U):
    """
    Compute the interval softmax probabilities q_L and q_U efficiently using PyTorch.

    Args:
        a_L (torch.Tensor): Lower bound tensor, shape (C,)
        a_U (torch.Tensor): Upper bound tensor, shape (C,)

    Returns:
        tuple: (q_L, q_U), interval softmax probabilities, each with shape (C,)
    """
    # Compute midpoints
    mid_points = (a_U + a_L) / 2

    # Compute denominators
    exp_a_L = torch.exp(a_L)
    exp_a_U = torch.exp(a_U)
    exp_mid_points = torch.exp(mid_points)

    sum_exp_mid_points = torch.sum(exp_mid_points, dim=-1, keepdim=True)

    denom_q_L = exp_a_L + (sum_exp_mid_points - exp_mid_points)
    denom_q_U = exp_a_U + (sum_exp_mid_points - exp_mid_points)

    # Compute numerators and final softmax
    q_L = exp_a_L / denom_q_L
    q_U = exp_a_U / denom_q_U

    return q_L, q_U

def actually_reachable(q_L, q_U):
    """
    Compute the actually reachable interval probabilities q_L_star and q_U_star.

    Args:
        q_L (torch.Tensor): Lower interval probabilities, shape (num_nodes, C) or (C,)
        q_U (torch.Tensor): Upper interval probabilities, shape (num_nodes, C) or (C,)

    Returns:
        tuple: (q_L_star, q_U_star), reachable interval probabilities, each of shape (num_nodes, C) or (C,)
    """
    sum_q_L = torch.sum(q_L, dim=-1, keepdim=True)
    sum_q_U = torch.sum(q_U, dim=-1, keepdim=True)

    q_L_star = torch.maximum(q_L, 1 - (sum_q_U - q_U))
    q_U_star = torch.minimum(q_U, 1 - (sum_q_L - q_L))

    return q_L_star, q_U_star

def cross_entropy_loss(predictions, targets, epsilon=1e-9):
    """
    Computes cross-entropy loss.

    Args:
        predictions (torch.Tensor): probability distribution (shape [C] or [num_nodes, C]).
        targets (torch.Tensor): one-hot encoded target (shape [C] or [num_nodes, C]).
        epsilon (float): small value to avoid log(0).

    Returns:
        torch.Tensor: scalar loss if input is a single sample, or tensor of losses if batched.
    """
    # Ensure tensors have the same shape
    assert predictions.shape == targets.shape, "Shapes of predictions and targets must match, but got {} and {}".format(predictions.shape, targets.shape)

    # Compute element-wise cross-entropy
    loss = -torch.sum(targets * torch.log(predictions + epsilon), dim=-1)

    return loss


def checker(q_L, q_U, a_L, a_U):
    """
    Check if the interval softmax probabilities are valid.

    Args:
        q_L (torch.Tensor): Lower bound probabilities.
        q_U (torch.Tensor): Upper bound probabilities.
        a_L (torch.Tensor): Lower bound logits.
        a_U (torch.Tensor): Upper bound logits.

    Raises:
        AssertionError: If the conditions are not met.
    """
    for i in range(q_L.shape[0]):
        assert torch.all(q_L[i] <= q_U[i]), f"Lower bounds must be less than or equal to upper bounds. Got {q_L[i]} and {q_U[i]}, a was {a_L[i]} and {a_U[i]}"
