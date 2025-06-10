import argparse
import json
import numpy as np
import random
from transformers import AdamW
from transformers import set_seed
import torch
import torch.nn.functional as F
import wandb

from utils.train_eval_utils import train, get_criterion
from data.dataloaders import get_dataloads
from models.model import all_model_names, all_base_pretrained_models, get_device, get_model

def set_all_seeds(seed):
    set_seed(seed)  # Hugging Face transformers
    torch.manual_seed(seed)  # PyTorch
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python random
    torch.cuda.manual_seed_all(seed)  # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiments(args):
    # Get and check the args
    model_name = args.model_name
    size = args.size
    validation = args.validation
    n_epochs = args.epochs
    seed = args.seed
    pretrained_model = args.pretrained_model_name
    learning_rate = args.lr
    weight_decay = args.wd    
    assert validation in [True, False], "Invalid validation setting: {}".format(validation)
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    assert size in ["small", "medium", "large"], "Invalid size setting: {}".format(size)
    print(f"Args: {model_name} using pretrained model {pretrained_model} with seed {seed} on {size} Point d'arrêt dataset with validation={validation}, for {n_epochs} epochs, a learning rate of {learning_rate} and weight decay of {weight_decay}...")

    device = get_device()

    # Init wandb
    wandb.init(project="pointdarret-classification", config=vars(args))

    # Set seeds
    set_all_seeds(seed)

    # Get the data loaders
    train_loader, test_loader, val_loader = get_dataloads(args.data_path, size, validation, args.test_size, tokenizer_name=pretrained_model, batch_size=args.batch_size, seed=seed)
    print("Training set size: ", len(train_loader))
    print("Validation set size: ", len(val_loader))
    print("Test set size: ", len(test_loader))
    
    # Instantiate the model
    model = get_model(args)

    # Define optimizer and loss function
    class_counts = train_loader['labels'].value_counts(sort=False).tolist()  
    print(f"class counts {class_counts}")
    criterion = get_criterion(device=device, balanced=False, class_counts=class_counts)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train
    train(args, model, pretrained_model, train_loader, val_loader, test_loader, criterion, optimizer, device=device)


def parse_args(parser):
    models_string = json.dumps(all_model_names)
    pretrained_model_string = json.dumps(all_base_pretrained_models)

    # Data args
    parser.add_argument("--data_path", type=str, default="data/clean_annotated_comments.csv")
    parser.add_argument("--output_dir", type=str, default="./camembertv2_results")
    parser.add_argument('--size', type=str, default='small', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')
    
    # Model args
    parser.add_argument("--model-name", type=str, default="text-class", help='the model to use, can take one of the following values: ' + models_string)
    parser.add_argument('--pretrained-model-name', type=str, default="almanach/camembertv2-base", help='name for pretrained text model to use to generate text embeddings, can take one of the following values: ' + pretrained_model_string)
    parser.add_argument("--attention-probs-dropout-prob", type=float, metavar="D", default=0.3, help="dropout probability for attention weights")
    parser.add_argument("--hidden-dropout-prob", type=float, metavar="D", default=0.3, help="dropout probability after hidden layer")
    
    # Hyper params
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=3, metavar='E', help='number of epochs')

    parser.add_argument("--seed", type=int, default=42)
    
    '''
    parser.add_argument('--undirected', type=bool, default=False, help='define the graph model as an undirected graph')
    parser.add_argument('--temp-edges', type=bool, default=False, help='add temporal edges to the graph')
    parser.add_argument('--num-layers', type=int, default=1, help='the number of GAT layers in graph models')
    parser.add_argument('--trim', type=str, default="reactions", help='graph construction trimming streatgy, should be either affordance, recent, or left empty for no trimming.')
    parser.add_argument('--new-trim', type=bool, default=False, help='rather or not to use the new trimming strategy (edge from post to target node only, instead of edges from post to all other nodes)')
    # Arguments related to dropout
    parser.add_argument("--dropout", type=float, metavar="D", default=0.4, help="dropout probability")
    '''

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for point d'arrêt detection")
    args = parse_args(parser)
    run_experiments(args)
