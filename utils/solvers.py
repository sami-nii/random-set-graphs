import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from tqdm import tqdm

def entropy(q):
    return -np.sum(q*np.log2(np.clip(q, 1e-12, np.max(q))))  # add small epsilon to avoid log(0)

def constraint_sum(q):
    return np.sum(q) - 1

def calculate_entropy_single(lb, ub, objective ):
    q0 = (ub + lb) /2

    bounds = [(lb[i], ub[i]) for i in range(len(lb))]
    constraints = [{'type': 'eq', 'fun': constraint_sum}]

    result = minimize(objective, q0, bounds=bounds, constraints=constraints, method='SLSQP')

    return -result.fun, result.x

def calculate_entropy(lower_bound, upper_bound, direction, delta=1e-5):
    """
    Maximizes or minimizes entropy for batched lower and upper bounds.

    Args:
        lower_bound (np.ndarray): Lower bounds array with shape (num_nodes, C).
        upper_bound (np.ndarray): Upper bounds array with shape (num_nodes, C).
        direction (str): "maximize" or "minimize".

    Returns:
        tuple: (optimal_probabilities, entropy_values) with shape (num_nodes, C) and (num_nodes).

    Raises:
        ValueError: If initial guess cannot be generated.
    """

    assert lower_bound.shape == upper_bound.shape, "Lower and upper bounds must have the same shape"
    assert direction in ["maximize", "minimize"], "Direction must be either 'maximize' or 'minimize'"
    assert np.all(lower_bound <= upper_bound + 1e-6), "Lower bounds must be less than or equal to upper bounds"
    assert len(lower_bound.shape) == 2, "Lower and upper bounds must be 2D arrays"

    upper_bound = np.clip(upper_bound, delta, np.max(upper_bound))

    num_nodes, C = lower_bound.shape
    optimal_probabilities = np.zeros((num_nodes, C))
    entropy_values = np.zeros(num_nodes)

    if direction == "maximize":
        objective = lambda q: -entropy(q)
    elif direction == "minimize":
        objective = entropy
    else:
        raise ValueError("Direction must be either 'maximize' or 'minimize'")

    for node_idx in tqdm(range(num_nodes), desc=f"Calculating entropy ({direction})", total=num_nodes):
        lb = lower_bound[node_idx]
        ub = upper_bound[node_idx]

        value, optimum = calculate_entropy_single(lb, ub, objective)

        entropy_values[node_idx] = value
        optimal_probabilities[node_idx] = optimum


    return optimal_probabilities, entropy_values


