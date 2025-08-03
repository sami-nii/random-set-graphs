import os
import re

def find_best_checkpoints(dataset_name: str, num_models: int) -> list[str]:
    """
    Finds the paths of the top-performing model checkpoints for a given dataset.
    Assumes the checkpoints are named in the format:
    {run_id}_{dataset_name}_val_f1={val_f1:.4f}.ckpt
    Assumes all checkpoints are stored in a single 'checkpoints' directory.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'Cora').
        num_models (int): The number of top models to retrieve (M).

    Returns:
        A list of file paths to the best checkpoints.
    """

    checkpoint_dir = "checkpoints"
    print(f"Searching for '{dataset_name}' checkpoints in: {checkpoint_dir}")

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Root checkpoint directory not found: '{checkpoint_dir}'. Please run training first.")

    checkpoints = []
    # Regex to find the val_f1 score in the filename
    score_pattern = re.compile(r"val_f1=([\d\.]+)\.ckpt")

    for filename in os.listdir(checkpoint_dir):
        ### CORRECTED: Filter files by dataset name ###
        # Check if the filename matches the dataset and is a checkpoint file
        if f"_{dataset_name}_" in filename and filename.endswith(".ckpt"):
            match = score_pattern.search(filename)
            if match:
                score = float(match.group(1))
                full_path = os.path.join(checkpoint_dir, filename)
                checkpoints.append((full_path, score))

    # Sort checkpoints by F1 score in descending order
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    if not checkpoints:
        raise FileNotFoundError(
            f"No valid checkpoints found for dataset '{dataset_name}' in {checkpoint_dir}. "
            f"Ensure filenames have the format '..._{dataset_name}_val_f1=...'.")

    # Select the top M models
    top_checkpoints = checkpoints[:num_models]
    
    if len(top_checkpoints) < num_models:
        print(f"Warning: Found only {len(top_checkpoints)} checkpoints for '{dataset_name}', but {num_models} were requested.")

    print(f"Found {len(top_checkpoints)} top models for '{dataset_name}' to form the ensemble:")
    for path, score in top_checkpoints:
        print(f"  - Path: {os.path.basename(path)}, Score: {score:.4f}")
        
    return [path for path, score in top_checkpoints]

def search_best_model(save_path, dataset_name):
    """
    Search for the best model in the given save path based on the dataset name.
    It assumes the models are saved in the format: {run_id}_{dataset_name}_val_f1={val_f1:.4f}.ckpt

    Args:
        save_path (str): The path where the models are saved.
        dataset_name (str): The name of the dataset to search for.

    Returns:
        str: The path to the best model file.
    """
    best_score = float('-inf')
    best_model_path = None

    # Regex pattern to extract val_f1 score
    pattern = re.compile(rf".+_{re.escape(dataset_name)}_val_f1=([0-9.]+)\.ckpt$")

    for filename in os.listdir(save_path):
        match = pattern.match(filename)
        if match:
            val_f1_str = match.group(1)
            try:
                val_f1 = float(val_f1_str)
                if val_f1 > best_score:
                    best_score = val_f1
                    best_model_path = os.path.join(save_path, filename)
            except ValueError:
                continue  # In case of malformed float string

    if best_model_path is None:
        raise FileNotFoundError(f"No valid model files found for dataset '{dataset_name}' in '{save_path}'")

    return best_model_path
