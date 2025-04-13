from utils.math import cross_entropy_loss
import torch


class CreNetLoss(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, q_L, q_U, target):
        """
        CreNet loss function
        Parameters:
        ----------
        q_L: torch.Tensor
            Lower bounds of the credal set, shape (num_nodes, C)
        q_U: torch.Tensor
            Upper bounds of the credal set, shape (num_nodes, C)
        target: torch.Tensor
            Target labels (one-hot encoded), shape (num_nodes, C)
        Returns:
        ----------
        torch.Tensor
            The loss value        
        """
        
        assert len(q_L.shape) == 2 and len(q_U.shape) == 2, f"q_L and q_U must be 2D tensors, but got shapes {q_L.shape} and {q_U.shape}"
        assert q_L.shape == q_U.shape, f"q_L and q_U must have the same shape, but got {q_L.shape} and {q_U.shape}"
        assert target.shape[0] == q_L.shape[0], f"target and q_L must have the same number of samples, but got {target.shape[0]} and {q_L.shape[0]}"
        assert target.shape[1] == q_L.shape[1], f"target and q_L must have the same number of classes, but got {target.shape[1]} and {q_L.shape[1]}"

        # Number of nodes
        num_nodes = q_L.shape[0]

        # Cross-entropy loss for q_U
        vanilla_component = cross_entropy_loss(q_U, target).mean()

        # Cross-entropy loss for q_L
        dro_component = cross_entropy_loss(q_L, target)

        # print(f'shape of dro_component: {dro_component.shape}, shape of target: {target.shape}, shape of q_L: {q_L.shape}')
        assert len(dro_component.shape) == 1, f"dro_component must be a 1D tensor, but got shape {dro_component.shape}"
        assert dro_component.shape[0] == num_nodes, f"dro_component must have the same number of samples as num_nodes, but got {dro_component.shape[0]} and {num_nodes}"

        # Select top delta * num_nodes cross-entropy values from dro_component
        top_values, _ = torch.topk(input=dro_component, k = max(1, int(self.delta * num_nodes)))

        dro_component = top_values.mean()

        # Total loss as the sum of both components
        total_loss = vanilla_component + dro_component

        return total_loss

