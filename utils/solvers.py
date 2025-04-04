import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog



def initial_guess(l, u, delta=1e-5):
    """
    Generates an initial guess for the optimization problem based on the lower and upper bounds and respectiong the normalization constraint.
    The initial guess is a uniform distribution within the bounds, ensuring that the sum of probabilities equals 1.
    Args:
        l (np.ndarray): Lower bounds for each class.
        u (np.ndarray): Upper bounds for each class.
        delta (float): Small value to avoid zero probabilities.
    Returns:
        np.ndarray: Initial guess for the optimization problem.
    """
    assert len(l) == len(u), "Lower and upper bounds must have the same length"
    assert np.all(l <= u), "Lower bounds must be less than or equal to upper bounds"

    n = len(l)
    c = np.zeros(n)
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    u_clip = np.clip(u, delta, np.max(u))

    bounds = [(l[i], u_clip[i]) for i in range(n)]

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    if res.success:
        return res.x
    else:
        return None



def calculate_entropy(lower_bound, upper_bound, direction):
    """
    Maximizes or minimizes entropy for batched lower and upper bounds.

    Args:
        lower_bound (np.ndarray): Lower bounds array with shape (num_nodes, C).
        upper_bound (np.ndarray): Upper bounds array with shape (num_nodes, C).
        direction (str): "maximize" or "minimize".

    Returns:
        tuple: (optimal_probabilities, entropy_values) both with shape (num_nodes, C) and (num_nodes,) respectively.

    Raises:
        ValueError: If initial guess cannot be generated.
    """

    assert lower_bound.shape == upper_bound.shape, "Lower and upper bounds must have the same shape"
    assert direction in ["maximize", "minimize"], "Direction must be either 'maximize' or 'minimize'"
    assert np.all(lower_bound <= upper_bound), "Lower bounds must be less than or equal to upper bounds"
    assert len(lower_bound.shape) == 2, "Lower and upper bounds must be 2D arrays"

    def entropy(q):
        return -np.sum(q * np.log2(q + 1e-12))  # add small epsilon to avoid log(0)

    def constraint_sum(q):
        return np.sum(q) - 1


    num_nodes, C = lower_bound.shape
    optimal_probabilities = np.zeros((num_nodes, C))
    entropy_values = np.zeros(num_nodes)

    for node_idx in range(num_nodes):
        lb = lower_bound[node_idx]
        ub = upper_bound[node_idx]

        q0 = initial_guess(lb, ub)
        if q0 is None:
            raise ValueError(f"Failed to generate an initial guess for node {node_idx}.")

        bounds = [(lb[i], ub[i]) for i in range(C)]
        constraints = [{'type': 'eq', 'fun': constraint_sum}]

        if direction == "maximize":
            result = minimize(lambda q: -entropy(q), q0, bounds=bounds, constraints=constraints)
            entropy_values[node_idx] = entropy(result.x)
        elif direction == "minimize":
            result = minimize(entropy, q0, bounds=bounds, constraints=constraints)
            entropy_values[node_idx] = entropy(result.x)
        else:
            raise ValueError("Direction must be either 'maximize' or 'minimize'")

        optimal_probabilities[node_idx] = result.x

    return optimal_probabilities, entropy_values


if __name__ == "__main__":


    # Example input vectors
    lower_bound = np.array([0.1, 0.1, 0.1])
    upper_bound = np.array([0.4, 0.2, 0.7])

    # Perform optimization
    optimal_q, optimal_entropy = calculate_entropy(lower_bound, upper_bound, "maximize")
    optimal_q, optimal_entropy

    print("Optimal q:", optimal_q)
    print("Optimal entropy:", optimal_entropy)
