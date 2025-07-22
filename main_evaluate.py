import argparse
import json
import numpy as np
import os
import pandas as pd
import random
from transformers import AdamW
from transformers import set_seed
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from utils.train_eval_utils import evaluate_model, inference_model, get_criterion
from data.dataloaders import get_dataloads
from models.model import all_model_names, all_base_pretrained_models, get_device, get_model
from transformers import get_scheduler

def set_all_seeds(seed):
    set_seed(seed)  # Hugging Face transformers
    torch.manual_seed(seed)  # PyTorch
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python random
    torch.cuda.manual_seed_all(seed)  # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_eval(args):
    # Get and check the args
    model_name = args.model_name
    size = args.size
    validation = args.validation
    n_epochs = args.epochs
    seed = args.seed
    pretrained_model = args.pretrained_model_name
    learning_rate = args.lr
    weight_decay = args.wd   
    loss_mino_class_weight = args.loss_minority_class_weight
    checkpoint_path = args.checkpoint_path
    inference_mode = args.inference_only

    assert validation in [True, False], "Invalid validation setting: {}".format(validation)
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    assert size in ["small", "medium", "large"], "Invalid size setting: {}".format(size)
    print(f"Args: {model_name} using pretrained model {pretrained_model} with seed {seed} on {size} Point d'arrêt dataset with validation={validation}, for {n_epochs} epochs, a learning rate of {learning_rate} and weight decay of {weight_decay}, batch sie if {args.batch_size}, loss minority class weight is {loss_mino_class_weight}...")


    print(f"ARGUMENTS: {args}")

    device = get_device()

    # Init wandb only on rank 0

    wandb.init(project="pointdarret-classification", config=vars(args))

    # Set seeds
    set_all_seeds(seed)

    # Get the data loaders
    # Use DistributedSampler for DDP
    _, test_loader, _, class_counts = get_dataloads(
        args.data_path,
        size,
        validation,
        args.test_size,
        tokenizer_name=pretrained_model,
        batch_size=args.batch_size,
        seed=seed,
        distributed=False,      
        model_name=model_name,
        inference_only=inference_mode,
    )

    print("Test set size: ", len(test_loader))

    # Define optimizer and loss function
    if loss_mino_class_weight < 0:
        class_counts = torch.tensor(class_counts) 
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
    else:
        class_weights = torch.tensor([1 - loss_mino_class_weight, loss_mino_class_weight])
    print(f"class weights {class_weights}")
    if args.loss == "focal":
        criterion = None #Not yet supported
    else:
        criterion = get_criterion(device=device, balanced=False, class_weights=class_weights)


    # Instantiate the model
    model = get_model(args).to(device)

    print("checkpoint_path:", checkpoint_path, type(checkpoint_path))


    # Load your checkpoint (adjust path as needed)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # If saved with DataParallel, keys will start with "module."
    # Create a new dict without "module." prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = value

    # Now load the fixed state dict into your model
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    if inference_mode:
        print("Running inference ...")
        pred_labels, pred_scores, pred_indices = inference_model(model, test_loader, model_name, device, threshold=args.threshold)

        # Read the original dataframe to preserve other columns
        df = pd.read_csv(args.data_path)
        df = df.reset_index(drop=True)
        df["pred_label"] = pd.Series(pred_labels, index=pred_indices)
        df["pred_score"] = pd.Series(pred_scores, index=pred_indices)
        df = df.sort_index()

        # Write output
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Inference completed. Results saved to {args.output_csv}")
        else:
            print("No output path provided. Use --output-csv to save predictions.")
        
    else: # Evaluation
        print("Running evaluation ...")
        test_loss, test_accuracy, test_f1, test_precision, test_recall, selected_threshold = evaluate_model(model, test_loader, model_name, device, f"{model_name}_{size}_test_outputs.tsv", tune_threshold=False, best_threshold=args.threshold, criterion=criterion)

        assert selected_threshold == args.threshold 

        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })
        print(f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, "
            f"Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}, Threshold: {selected_threshold:.4f}")

    # Finish the run
    wandb.finish()



def parse_args(parser):
    models_string = json.dumps(all_model_names)
    pretrained_model_string = json.dumps(all_base_pretrained_models)

    # Data args
    parser.add_argument("--data_path", type=str, default="data/clean_annotated_comments.csv")
    parser.add_argument("--checkpoint_path", type=str, default="/home/cnouri/Fb-points-darret/models/checkpoints/20250703_183140_almanach/camembert-base_large.pt", help='path to the model checkpoint file')

    parser.add_argument("--output_dir", type=str, default="./camembertv2_results")
    parser.add_argument('--size', type=str, default='large', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    parser.add_argument("--loss-minority-class-weight", type=float, default=-1, help='cross entropy loss weight applied to the minority class, if negative, then the class weight is computed using the train set class distribution')

    # Model args
    parser.add_argument("--model-name", type=str, default="text_only", help='the model to use, can take one of the following values: ' + models_string)
    parser.add_argument('--pretrained-model-name', type=str, default="almanach/camembert-base", help='name for pretrained text model to use to generate text embeddings, can take one of the following values: ' + pretrained_model_string)
    parser.add_argument("--attention-probs-dropout-prob", type=float, metavar="D", default=0.3, help="dropout probability for attention weights")
    parser.add_argument("--hidden-dropout-prob", type=float, metavar="D", default=0.3, help="dropout probability after hidden layer")
    parser.add_argument("--loss", type=str, default="crossentropy", help='loss can be: focal, crossentropy ...')
    parser.add_argument("--threshold", type=float, default=0.9, help='best threshold value determined during training on the validation set')

    
    # Hyper params
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')
    parser.add_argument("--seed", type=int, default=42)

    # Inference argument
    parser.add_argument('--inference-only', action='store_true', help='Run inference only and write predictions to CSV')
    parser.add_argument('--output-csv', type=str, default=None, help='Path to save inference results (CSV with predicted scores and labels)')

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for point d'arrêt detection")
    args = parse_args(parser)
    run_eval(args)

