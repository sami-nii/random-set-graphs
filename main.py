import wandb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trainers
import argparse
from functools import partial
import importlib
from utils.model_manager import search_best_model


# Set the dataset name with argument parsing
parser = argparse.ArgumentParser(description="Run a sweep for a specific dataset.")

parser.add_argument(
    "-d", 
    "--dataset",
    type=str,
    choices=["chameleon", "patents", "arxiv", "reddit2", "coauthor"],
    default="squirrel",
    help="Dataset to run the sweep on.",
)

parser.add_argument(
    "-s",
    "--sweep",
    type=str,
    default="",
    help="sweep id to run",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    choices=["vanilla", "credal"],
    default="vanilla",
    help="Model to run the sweep on.",
)

parser.add_argument(
    "-p",
    "--project_name",
    type=str,
    default="graph-uncertainty",
    help="Project name for wandb.",
)

parser.add_argument(
    "--save_path",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "checkpoints"),
    help="Path to save the model checkpoints.",
)

args = parser.parse_args()

try:
    # Dynamically construct the name of the sweep variable
    sweep_var_name = f"sweep_{args.dataset}_{args.model}"
    # Dynamically import the sweeps module
    sweeps_module = importlib.import_module("sweeps.sweeps")
    # Get the sweep dict by name
    sweep = getattr(sweeps_module, sweep_var_name)
except AttributeError:
    raise ValueError(f"No sweep found for combination: {sweep_var_name}")


if args.model == "vanilla":
    train_func = trainers.vanilla_train
elif args.model == "odin":
    train_func = trainers.odin_test
elif args.model == "credal":
    train_func = trainers.credal_train
else:
    raise ValueError(f"Unsupported model: {args.model}")


save_path = None

if  args.model == "odin": # odin is post-hoc, so it needs a pre-trained model (vanilla)
    # the save path should be the best model for the selected dataset
    save_path = search_best_model(args.save_path, args.dataset)

if args.model == "vanilla": # save the pre-trained model
    # the save path should be the folder where the models will be saved
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path


sweep["name"] = f"{args.dataset}_{args.model}"


if args.sweep:
    sweep_id = args.sweep
else:
    sweep_id = wandb.sweep(sweep=sweep, project=args.project_name)


train_func = partial(
    train_func, 
    project_name=args.project_name, 
    dataset_name=args.dataset, 
    save_path=save_path
)

print(f"Running sweep with ID: {sweep_id}")
wandb.agent(sweep_id, function=train_func, project=args.project_name)

    
    
