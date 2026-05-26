"""
train_target.py
───────────────
Training entry point for the 6-class target classification task.
Mirrors train.py but imports exclusively from:
  • data/dataloaders_target.py
  • utils/train_eval_utils_target.py
"""

import argparse
import json
import numpy as np
import os
import random

from torch.optim import AdamW
from transformers import set_seed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from utils.train_eval_utils_target import get_criterion, train
from data.dataloaders_target import get_dataloads, NUM_LABELS, TARGET_CLASSES
from models.model import all_model_names, all_base_pretrained_models, get_device, get_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Focal loss (multi-class)
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    class_counts : 1-D tensor of length NUM_LABELS (one count per class).
    """
    def __init__(self, class_counts: torch.Tensor, device,
                 alpha: float = 0.5, gamma: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        self.weight = class_weights.to(device)
        print(
            f"FocalLoss | alpha={alpha}, gamma={gamma} | "
            f"per-class weights: {dict(zip(TARGET_CLASSES, class_weights.tolist()))}"
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  Distributed helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", init_method="env://",
                            rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def set_all_seeds(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════════════════════════
#  Main experiment function (one DDP process)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiments(rank: int, world_size: int, args):
    setup(rank, world_size)
    device = get_device(rank=rank)

    if rank == 0:
        wandb.init(project="target-classification", config=vars(args))
        print(f"=== Target Classification | {NUM_LABELS} classes ===")
        print(f"Classes: {TARGET_CLASSES}")
        print(f"Args: {args}")

    set_all_seeds(args.seed)

    # ── Data ───────────────────────────────────────────────────────────────────
    (train_loader, test_loader, val_loader,
     class_counts, class_weights, label2id, id2label) = get_dataloads(
        args.data_path,
        size=args.size,
        validation=args.validation,
        test_size=args.test_size,
        tokenizer_name=args.pretrained_model_name,
        batch_size=args.batch_size,
        distributed=True,
        rank=rank,
        world_size=world_size,
        model_name=args.model_name,
        balance_strategy=args.balance_strategy,
    )

    if rank == 0:
        print(f"Train batches: {len(train_loader)}")
        if args.validation and val_loader:
            print(f"Val batches:   {len(val_loader)}")
        print(f"Test batches:  {len(test_loader)}")
        print(f"Class counts (after balancing): {dict(zip(TARGET_CLASSES, class_counts))}")

    # ── Model ──────────────────────────────────────────────────────────────────
    args.num_classes = NUM_LABELS   # consumed by get_model()
    model = get_model(args).to(device)
    model = DDP(model, device_ids=[rank])

    # ── Loss ───────────────────────────────────────────────────────────────────
    if args.loss == "focal":
        counts_tensor = torch.tensor(class_counts, dtype=torch.float)
        criterion = FocalLoss(counts_tensor, device,
                              alpha=args.focal_alpha,
                              gamma=args.focal_gamma)
    else:
        # class_weights already inverse-frequency balanced from the dataloader
        criterion = get_criterion(device=device, class_weights=class_weights)

    # ── Optimiser ──────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── Train ──────────────────────────────────────────────────────────────────
    train(
        args, model, args.pretrained_model_name,
        train_loader, val_loader, test_loader,
        criterion, optimizer,
        device=device, rank=rank,
        id2label=id2label,
    )

    cleanup()


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    models_str     = json.dumps(all_model_names)
    pretrained_str = json.dumps(all_base_pretrained_models)

    # ── Data ───────────────────────────────────────────────────────────────────
    parser.add_argument("--data_path", type=str,
                        default="data/clean_annotated_comments.csv")
    parser.add_argument("--output_dir", type=str, default="./target_results")
    parser.add_argument("--size", type=str, default="large",
                        choices=["small", "medium", "large"],
                        help="Dataset size to use")
    parser.add_argument("--validation", type=bool, default=True,
                        help="Whether to use a validation split")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data held out for testing")
    parser.add_argument("--balance-strategy", type=str, default="oversample",
                        choices=["oversample", "undersample", "none"],
                        help="Class balancing applied to the training set")

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model-name", type=str, default="text_only",
                        help="Model architecture. One of: " + models_str)
    parser.add_argument("--pretrained-model-name", type=str,
                        default="almanach/camembert-base",
                        help="Pretrained encoder. One of: " + pretrained_str)
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.3)
    parser.add_argument("--hidden-dropout-prob", type=float, default=0.3)

    # ── Loss ───────────────────────────────────────────────────────────────────
    parser.add_argument("--loss", type=str, default="crossentropy",
                        choices=["crossentropy", "focal"],
                        help="Loss function")
    parser.add_argument("--focal-alpha", type=float, default=0.5,
                        help="Alpha for FocalLoss (used only if --loss focal)")
    parser.add_argument("--focal-gamma", type=float, default=0.5,
                        help="Gamma for FocalLoss (used only if --loss focal)")

    # ── Hyper-params ───────────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5, metavar="E")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run_ddp_training(args):
    world_size = torch.cuda.device_count()
    mp.spawn(run_experiments, args=(world_size, args),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a 6-class target classifier on moderation data"
    )
    args = parse_args(parser)
    run_ddp_training(args)