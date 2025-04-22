import os
import re

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
