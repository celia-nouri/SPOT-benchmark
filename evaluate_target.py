"""
evaluate_target.py
──────────────────
Evaluation and inference script for the 6-class target classifier.
Mirrors evaluate.py but uses:
  • data/dataloaders_target.py
  • utils/train_eval_utils_target.py
"""

import argparse
import json
import numpy as np
import os
import pandas as pd
import random

from transformers import set_seed
import torch
import torch.nn as nn
import wandb

from utils.train_eval_utils_target import evaluate_model, inference_model, get_criterion
from data.dataloaders_target import get_dataloads, NUM_LABELS, TARGET_CLASSES, LABEL2ID, ID2LABEL
from models.model import all_model_names, all_base_pretrained_models, get_device, get_model


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed: int):
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(model, checkpoint_path: str, device):
    """
    Load a checkpoint saved with DDP (keys may start with 'module.').
    Uses strict=False to tolerate minor architecture mismatches.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    clean = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean, strict=False)
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(args):
    # ── Validation ─────────────────────────────────────────────────────────────
    assert args.model_name in all_model_names, f"Invalid model name: {args.model_name}"
    assert args.size in ["small", "medium", "large"], f"Invalid size: {args.size}"
    assert not args.inference_only or args.output_csv, \
        "--output-csv is required when running in --inference-only mode"

    print(f"Args: {args}")

    device = get_device()
    set_all_seeds(args.seed)

    wandb.init(project="target-classification", config=vars(args))

    # ── Data ───────────────────────────────────────────────────────────────────
    # We only need the test loader; train/val loaders are discarded.
    _, test_loader, _, class_counts, class_weights, label2id, id2label = get_dataloads(
        args.data_path,
        size=args.size,
        validation=False,           # no val split needed at eval time
        test_size=args.test_size,
        tokenizer_name=args.pretrained_model_name,
        batch_size=args.batch_size,
        seed=args.seed,
        distributed=False,
        model_name=args.model_name,
        inference_only=args.inference_only,
        tokenize=True,
    )
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {TARGET_CLASSES}")

    # ── Model ──────────────────────────────────────────────────────────────────
    args.num_classes = NUM_LABELS
    model = get_model(args)
    model = load_checkpoint(model, args.checkpoint_path, device)
    model.eval()

    # ── Loss (needed for eval mode only) ───────────────────────────────────────
    criterion = None
    if not args.inference_only:
        criterion = get_criterion(device=device, class_weights=class_weights)

    # ══════════════════════════════════════════════════════════════════════════
    #  INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    if args.inference_only:
        print("Running inference ...")

        pred_label_ids, pred_probs, pred_label_names, pred_indices = inference_model(
            model, test_loader, args.model_name, device, id2label=id2label
        )

        # Attach predictions to the original dataframe rows via stored indices
        df = pd.read_csv(args.data_path)
        df = df.reset_index(drop=True)
        df["index"] = range(len(df))

        df.loc[pred_indices, "pred_target_id"]   = pred_label_ids
        df.loc[pred_indices, "pred_target_label"] = pred_label_names

        # One column per class with its softmax probability
        for class_idx, class_name in id2label.items():
            col = f"pred_prob_{class_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            df.loc[pred_indices, col] = [p[class_idx] for p in pred_probs]

        df.to_csv(args.output_csv, index=False)
        print(f"Inference done. Results saved to {args.output_csv}")

    # ══════════════════════════════════════════════════════════════════════════
    #  EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    else:
        print("Running evaluation ...")

        metrics, predictions = evaluate_model(
            model, test_loader, args.model_name, device,
            output_file=f"{args.model_name}_{args.size}_test.json",
            criterion=criterion,
            id2label=id2label,
        )

        # ── Wandb logging ──────────────────────────────────────────────────────
        target_names = [id2label[i] for i in sorted(id2label)]
        per_class_log = {
            f"test_f1_{target_names[i]}": v
            for i, v in enumerate(metrics["per_class_f1"])
        }
        wandb.log({
            "test_loss":             metrics["loss"],
            "test_accuracy":         metrics["accuracy"],
            "test_macro_f1":         metrics["macro_f1"],
            "test_weighted_f1":      metrics["weighted_f1"],
            "test_macro_precision":  metrics["macro_precision"],
            "test_macro_recall":     metrics["macro_recall"],
            **per_class_log,
        })

        print(
            f"\nTest results\n"
            f"  Loss:          {metrics['loss']:.4f}\n"
            f"  Accuracy:      {metrics['accuracy']:.4f}\n"
            f"  Macro F1:      {metrics['macro_f1']:.4f}\n"
            f"  Weighted F1:   {metrics['weighted_f1']:.4f}\n"
            f"  Macro P:       {metrics['macro_precision']:.4f}\n"
            f"  Macro R:       {metrics['macro_recall']:.4f}\n"
            f"  Per-class F1:  { {target_names[i]: round(v, 4) for i, v in enumerate(metrics['per_class_f1'])} }"
        )

        # ── Error-analysis CSV ─────────────────────────────────────────────────
        if args.output_csv:
            df = pd.read_csv(args.data_path)
            df = df.reset_index(drop=True)
            df["index"] = range(len(df))

            indices = predictions["indices"]

            df.loc[indices, "true_target_id"]    = predictions["true_label_ids"]
            df.loc[indices, "pred_target_id"]    = predictions["pred_label_ids"]
            df.loc[indices, "true_target_label"] = [id2label[i] for i in predictions["true_label_ids"]]
            df.loc[indices, "pred_target_label"] = [id2label[i] for i in predictions["pred_label_ids"]]
            df.loc[indices, "correct"]           = [
                t == p for t, p in zip(predictions["true_label_ids"], predictions["pred_label_ids"])
            ]

            # One softmax-probability column per class
            for class_idx, class_name in id2label.items():
                col = f"prob_{class_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                df.loc[indices, col] = [row[class_idx] for row in predictions["pred_probs"]]

            # Confidence = probability assigned to the predicted class
            df.loc[indices, "pred_confidence"] = [
                row[pred] for row, pred in zip(predictions["pred_probs"], predictions["pred_label_ids"])
            ]

            df_test = df.loc[indices].copy()
            df_test.to_csv(args.output_csv, index=False)
            print(f"Error-analysis CSV saved to {args.output_csv} ({len(df_test)} rows)")

            # Quick summary
            n_correct = df_test["correct"].sum()
            print(f"  Correct: {n_correct}/{len(df_test)} "
                  f"({100*n_correct/len(df_test):.1f}%)")

    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
#  Argument parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    models_str     = json.dumps(all_model_names)
    pretrained_str = json.dumps(all_base_pretrained_models)

    # ── Data ───────────────────────────────────────────────────────────────────
    parser.add_argument("--data_path", type=str,
                        default="data/targets/stop_target.csv")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the .pt checkpoint saved by train_target.py")
    parser.add_argument("--output_dir", type=str, default="./target_results")
    parser.add_argument("--size", type=str, default="large",
                        choices=["small", "medium", "large"])
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data used as test set (must match training split)")

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument("--model-name", type=str, default="text_only",
                        help="Model architecture. One of: " + models_str)
    parser.add_argument("--pretrained-model-name", type=str,
                        default="almanach/camembert-base",
                        help="Pretrained encoder. One of: " + pretrained_str)
    parser.add_argument("--attention-probs-dropout-prob", type=float, default=0.3)
    parser.add_argument("--hidden-dropout-prob",          type=float, default=0.3)

    # ── Loss (eval mode only) ──────────────────────────────────────────────────
    parser.add_argument("--loss", type=str, default="crossentropy",
                        choices=["crossentropy", "focal"])

    # ── Runtime ────────────────────────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--validation", type=bool, default=False)  # unused at eval time
    parser.add_argument("--epochs",     type=int,  default=0)      # unused at eval time

    # ── Inference mode ─────────────────────────────────────────────────────────
    parser.add_argument("--inference-only", action="store_true",
                        help="Run inference only (no ground-truth labels required)")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Path to save predictions as CSV. "
                             "In --inference-only mode: required. "
                             "In eval mode: optional; produces an error-analysis CSV "
                             "with true/pred labels, per-class probabilities, and a 'correct' flag.")

    return parser.parse_args()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate or run inference with a 6-class target classifier"
    )
    args = parse_args(parser)
    run_eval(args)