import argparse
import json
import numpy as np
import os
import random
from torch.optim import AdamW
from transformers import set_seed
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from utils.train_eval_utils import train, get_criterion
from data.dataloaders import get_dataloads
from models.model import all_model_names, all_base_pretrained_models, get_device, get_model
from transformers import get_scheduler


class FocalLoss(nn.Module):
    def __init__(self, class_counts, device, alpha=0.5, gamma=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() 
        self.weight = class_weights.to(device)
        self.device = device
        print(f"Using Focal Loss with alpha {alpha}, gamma {gamma}, class weights {class_weights}")

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)  # softmax prob of the true class
        loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return loss.mean()

class FocalLoss2(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduction='mean'):
        super(FocalLoss2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()


def setup(rank, world_size):
    # Set required environment variables for env://
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def set_all_seeds(seed):
    set_seed(seed)  # Hugging Face transformers
    torch.manual_seed(seed)  # PyTorch
    np.random.seed(seed)  # NumPy
    random.seed(seed)  # Python random
    torch.cuda.manual_seed_all(seed)  # CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiments(rank, world_size, args):
    setup(rank, world_size)
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

    assert validation in [True, False], "Invalid validation setting: {}".format(validation)
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    assert size in ["small", "medium", "large"], "Invalid size setting: {}".format(size)
    print(f"Args: {model_name} using pretrained model {pretrained_model} with seed {seed} on {size} Point d'arrêt dataset with validation={validation}, for {n_epochs} epochs, a learning rate of {learning_rate} and weight decay of {weight_decay}, batch sie if {args.batch_size}, loss minority class weight is {loss_mino_class_weight}...")
    print(f"Distributed settings: Rank is {rank}, world size is {world_size}")


    print(f"ARGUMENTS: {args}")

    device = get_device(rank=rank)

    # Init wandb only on rank 0
    if rank == 0:
        wandb.init(project="pointdarret-classification", config=vars(args))

    # Set seeds
    set_all_seeds(seed)

    # Get the data loaders
    # Use DistributedSampler for DDP
    train_loader, test_loader, val_loader, class_counts = get_dataloads(
        args.data_path,
        size,
        validation,
        args.test_size,
        tokenizer_name=pretrained_model,
        batch_size=args.batch_size,
        seed=seed,
        distributed=True,      
        rank=rank,
        world_size=world_size,
        model_name=model_name,
    )

    if rank == 0:
        print("Training set size: ", len(train_loader))
        if validation:
            print("Validation set size: ", len(val_loader))
        print("Test set size: ", len(test_loader))

    # Instantiate the model
    model = get_model(args).to(device)
    model = DDP(model, device_ids=[rank])


    # Define optimizer and loss function
    if loss_mino_class_weight < 0:
        class_counts = torch.tensor(class_counts) 
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
    else:
        class_weights = torch.tensor([1 - loss_mino_class_weight, loss_mino_class_weight])
    print(f"class weights {class_weights}")
    if args.loss == "focal":
        criterion = FocalLoss(class_counts, device)
    else:
        criterion = get_criterion(device=device, balanced=False, class_weights=class_weights)

    #lr_scheduler = get_scheduler(
    #    name="linear",
    #    optimizer=optimizer,
    #    num_warmup_steps=100,
    #    num_training_steps=total_steps
    #)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Train
    train(args, model, pretrained_model, train_loader, val_loader, test_loader, criterion, optimizer, device=device, rank=rank)

    cleanup()


def parse_args(parser):
    models_string = json.dumps(all_model_names)
    pretrained_model_string = json.dumps(all_base_pretrained_models)

    # Data args
    parser.add_argument("--data_path", type=str, default="data/clean_annotated_comments.csv")
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

    
    # Hyper params
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=2, metavar='E', help='number of epochs')

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

def run_ddp_training(args):
    world_size = torch.cuda.device_count()
    mp.spawn(run_experiments, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for point d'arrêt detection")
    args = parse_args(parser)
    run_ddp_training(args)
