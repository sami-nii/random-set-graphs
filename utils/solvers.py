import numpy as np
from scipy.optimize import minimize, linprog
# from scipy.optimize import linprog # Not used in the provided code
from tqdm import tqdm
import multiprocessing
import ctypes
import os # To get cpu count


# Define shared memory array wrappers for easier handling
# Global variables to hold shared memory references within worker processes
shared_lower_bound = None
shared_upper_bound = None
shared_optimal_probabilities = None
shared_entropy_values = None
shared_params = {} # To hold shape, C, objective etc.

def entropy(q):
    # Use np.maximum instead of clip for slightly better performance sometimes
    q_clipped = np.maximum(q, 1e-12)
    return -np.sum(q * np.log2(q_clipped))

def constraint_sum(q):
    return np.sum(q) - 1

def initial_guess(l, u, delta=1e-5):
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

def min_max_entropy(l, u, x0, delta=1e-5):
    n = len(l)
    u_clip = np.clip(u, delta, np.max(u))

    bounds = [(l[i], u_clip[i]) for i in range(n)]
    # from scipy.optimize import maximize
    # Objective function
    def objective_min(x):
        return -np.sum(x*np.log2(np.clip(x, 1e-12, np.max(x))))
    def objective_max(x):
        return np.sum(x*np.log2(np.clip(x, 1e-12, np.max(x))))
    # Constraint function
    def constraint(x):
        return np.sum(x) - 1

    problem = {'type': 'eq', 'fun': constraint}
    sol1 = minimize(objective_min, x0, method='SLSQP', bounds=bounds, constraints=problem)
    sol2 = minimize(objective_max, x0, method='SLSQP', bounds=bounds, constraints=problem)
    return sol1.fun, -sol2.fun

# Function to initialize shared memory arrays in each worker process
def init_worker(lower_bound_base, upper_bound_base, min_entropy_vals_base, max_entropy_val_base, params):
    global shared_lower_bound, shared_upper_bound
    global min_shared_entropy_values, max_shared_entropy_values
    global shared_params

    shared_params.update(params) # Store shape, C, objective etc.
    num_nodes, C = shared_params['shape']

    # Create numpy arrays from the shared memory buffer without copying
    # These arrays share the underlying memory buffer
    shared_lower_bound = np.frombuffer(lower_bound_base.get_obj(), dtype=np.float64).reshape(num_nodes, C)
    shared_upper_bound = np.frombuffer(upper_bound_base.get_obj(), dtype=np.float64).reshape(num_nodes, C)
    min_shared_entropy_values = np.frombuffer(min_entropy_vals_base.get_obj(), dtype=np.float64)
    max_shared_entropy_values = np.frombuffer(max_entropy_val_base.get_obj(), dtype=np.float64)



# The task performed by each worker process
def worker_task(node_idx):
    global shared_lower_bound, shared_upper_bound
    global shared_optimal_probabilities, min_shared_entropy_values, max_shared_entropy_values
    global shared_params

    lb = shared_lower_bound[node_idx]
    ub = shared_upper_bound[node_idx]

    if not np.all(lb <= ub):
        print(f"Warning: received upper bound {ub} less than lower bound {lb} for node {node_idx}. Clipping lower bound to be equal to the upper bound .")
        lb = np.minimum(lb, ub) # Ensure lb <= ub, this could not be guaranteed due to numerical errors

    # --- Optimization Logic (same as calculate_entropy_single) ---
    q0 = initial_guess(lb, ub)

    min_entropy, max_entropy = min_max_entropy(lb, ub, q0)
    min_shared_entropy_values[node_idx] = min_entropy
    max_shared_entropy_values[node_idx] = max_entropy

    # No need to return anything as results are written to shared memory
    return None # Or return node_idx, result.success for progress tracking if needed


def calculate_entropy(lower_bound, upper_bound, delta=1e-5, num_workers=None):
    """
    Calculates the minimum and maximum entropy for batched lower and upper bounds using parallel processing.

    Args:
        lower_bound (np.ndarray): Lower bounds array with shape (num_nodes, C).
        upper_bound (np.ndarray): Upper bounds array with shape (num_nodes, C).
        delta (float): Small epsilon for clipping upper bounds away from zero.
        num_workers (int, optional): Number of worker processes. Defaults to the number of CPU cores.

    Returns:
        tuple: (min_entropy_values, max_entropy_values) with shape (num_nodes,) each.

    Raises:
        AssertionError: If input bounds are invalid or have mismatched shapes.
    """

    assert lower_bound.shape == upper_bound.shape, "Lower and upper bounds must have the same shape"
    # Relaxing this slightly for numerical stability post-clipping
    assert np.all(lower_bound <= upper_bound + 1e-6), "Lower bounds must be less than or equal to upper bounds"
    assert len(lower_bound.shape) == 2, "Lower and upper bounds must be 2D arrays"
    # Ensure bounds sum correctly (at least feasible)
    assert np.all(np.sum(lower_bound, axis=1) <= 1.0 + 1e-6), "Sum of lower bounds for a node cannot exceed 1"
    assert np.all(np.sum(upper_bound, axis=1) >= 1.0 - 1e-6), "Sum of upper bounds for a node must be at least 1"


    # Clip upper bound *before* putting into shared memory if needed
    # Note: Clipping lower bound might make problem infeasible if sum(lb) > 1
    # Consider if delta clipping is truly needed or if bounds are guaranteed >= 0
    # upper_bound = np.clip(upper_bound, delta, np.max(upper_bound)) # Original clipping
    # A potentially safer clip, only ensuring bounds are non-negative

    num_nodes, C = lower_bound.shape

    if num_workers is None:
        num_workers = os.cpu_count()
        print(f"Using {num_workers} worker processes.")

    # --- Create Shared Memory Arrays ---
    # Use double precision floats (np.float64 -> ctypes.c_double)
    lower_bound_base = multiprocessing.Array(ctypes.c_double, num_nodes * C)
    upper_bound_base = multiprocessing.Array(ctypes.c_double, num_nodes * C)
    min_entropy_vals_base = multiprocessing.Array(ctypes.c_double, num_nodes)
    max_entropy_vals_base = multiprocessing.Array(ctypes.c_double, num_nodes)

    # --- Wrap shared arrays as numpy arrays (for easy copying) ---
    # This creates temporary numpy views, data is then copied into shared memory
    lower_bound_shared_np = np.frombuffer(lower_bound_base.get_obj(), dtype=np.float64).reshape(num_nodes, C)
    upper_bound_shared_np = np.frombuffer(upper_bound_base.get_obj(), dtype=np.float64).reshape(num_nodes, C)

    # --- Copy data into shared memory ---
    np.copyto(lower_bound_shared_np, lower_bound)
    np.copyto(upper_bound_shared_np, upper_bound)

    # --- Prepare initializer arguments ---
    params = {'shape': (num_nodes, C)}
    initargs = (lower_bound_base, upper_bound_base, min_entropy_vals_base, max_entropy_vals_base, params)

    # --- Create and run the process pool ---
    # Use context manager for proper cleanup
    results = []
    with multiprocessing.Pool(processes=num_workers, initializer=init_worker, initargs=initargs) as pool:
        # pool.map will distribute node indices (0 to num_nodes-1) to worker_task
        # Wrap the range with tqdm for progress bar
        # Use imap_unordered for potential slight performance gain if order doesn't matter
        # for the progress bar, but map is fine and simpler. Chunksize can help.
        chunksize = max(1, num_nodes // (num_workers * 4)) # Heuristic chunksize
        list(tqdm(pool.imap_unordered(worker_task, range(num_nodes), chunksize=chunksize),
                  total=num_nodes, desc=f"Calculating entropy"))

        # pool.map(worker_task, range(num_nodes)) # Alternative without tqdm

    # --- Retrieve results from shared memory ---
    # Create numpy arrays viewing the final shared memory buffers
    # Important: Create copies if you want to release the shared memory later
    # or if the caller shouldn't rely on shared memory.
    min_entropy_values = np.frombuffer(min_entropy_vals_base.get_obj(), dtype=np.float64).copy()
    max_entropy_values = np.frombuffer(max_entropy_vals_base.get_obj(), dtype=np.float64).copy()

    del lower_bound_base, upper_bound_base, min_entropy_vals_base, max_entropy_vals_base

    return  min_entropy_values, max_entropy_values
