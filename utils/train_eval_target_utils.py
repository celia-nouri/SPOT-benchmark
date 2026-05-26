"""
train_eval_utils_target.py
──────────────────────────
Train / evaluation utilities for the 6-class target classification task.
Drop-in replacement for train_eval_utils.py when --task multiclass is used.
"""

from collections import Counter
from datetime import datetime
import json
import os

from models.model import all_model_names
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from types import SimpleNamespace
import wandb


# ═══════════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════════

def get_criterion(device, class_weights: torch.Tensor | None = None) -> nn.CrossEntropyLoss:
    """
    Weighted CrossEntropyLoss for multi-class classification.

    Parameters
    ----------
    class_weights : torch.Tensor of shape (num_classes,) | None
        Inverse-frequency weights returned by the dataloader.
        Pass None to use uniform weights.
    """
    if class_weights is not None:
        print(f"CrossEntropyLoss with class weights: {class_weights.tolist()}")
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    print("CrossEntropyLoss with uniform weights")
    return nn.CrossEntropyLoss()


# ═══════════════════════════════════════════════════════════════════════════════
#  Serialisation helper
# ═══════════════════════════════════════════════════════════════════════════════

def to_serializable_list(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy().tolist()
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [to_serializable_list(x) for x in data]
    return data


# ═══════════════════════════════════════════════════════════════════════════════
#  Model forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def run_model_pred(model, batch, model_name: str, tokenizer=None) -> SimpleNamespace:
    """
    Unified forward pass that returns a SimpleNamespace with a .logits attribute
    regardless of model architecture.
    """
    if model_name == "text_only" or "_concat" in model_name:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=None,
        )
        return outputs  # HuggingFace ModelOutput already has .logits

    elif "_embed" in model_name or model_name == "graph_context_all":
        logits = model(batch)
        return SimpleNamespace(logits=logits)

    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported: text_only, *_concat, *_embed, graph_context_all"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _predictions_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Argmax over class dimension → predicted label indices."""
    return torch.argmax(logits, dim=1)


def _compute_metrics(true_labels: list, pred_labels: list,
                     id2label: dict | None = None) -> dict:
    """
    Compute all multi-class metrics in one place.

    Returns a dict with keys:
        accuracy, macro_f1, weighted_f1, macro_precision, macro_recall,
        per_class_f1  (list, one entry per class)
    """
    target_names = [id2label[i] for i in sorted(id2label)] if id2label else None

    macro_f1      = f1_score(true_labels, pred_labels, average="macro",    zero_division=0)
    weighted_f1   = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    per_class_f1  = f1_score(true_labels, pred_labels, average=None,       zero_division=0).tolist()
    macro_prec    = precision_score(true_labels, pred_labels, average="macro",    zero_division=0)
    macro_rec     = recall_score   (true_labels, pred_labels, average="macro",    zero_division=0)
    accuracy      = float(sum(t == p for t, p in zip(true_labels, pred_labels))) / max(len(true_labels), 1)

    print(classification_report(true_labels, pred_labels,
                                 target_names=target_names, digits=4, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(true_labels, pred_labels))
    print("Predicted distribution:", pd.Series(pred_labels).value_counts().to_dict())

    return dict(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        macro_precision=macro_prec,
        macro_recall=macro_rec,
        per_class_f1=per_class_f1,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(
    model,
    loader,
    model_name: str,
    device,
    output_file: str = "",
    criterion: nn.Module | None = None,
    id2label: dict | None = None,
) -> dict:
    """
    Evaluate the model on a DataLoader and return a metrics dict.

    Returns
    -------
    dict with keys:
        loss, accuracy, macro_f1, weighted_f1,
        macro_precision, macro_recall, per_class_f1
    """
    model.eval()
    running_loss = 0.0
    true_labels, pred_labels, logits_list, indices_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = run_model_pred(model, batch, model_name)
            logits = outputs.logits
            labels = batch["labels"]

            if criterion is not None:
                running_loss += criterion(logits, labels).item()

            preds = _predictions_from_logits(logits)
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
            logits_list.append(logits.detach().cpu())
            if "index" in batch:
                indices_list.extend(batch["index"].cpu().tolist())

    metrics = _compute_metrics(true_labels, pred_labels, id2label)
    metrics["loss"] = running_loss / max(len(true_labels), 1)

    if output_file:
        out = {
            "pred_labels": to_serializable_list(pred_labels),
            "true_labels": to_serializable_list(true_labels),
            "logits": to_serializable_list(torch.cat(logits_list).numpy()),
        }
        if indices_list:
            out["index"] = indices_list
        with open(output_file, "w") as f:
            json.dump(out, f)
            f.write("\n")

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Inference (no ground truth required)
# ═══════════════════════════════════════════════════════════════════════════════

def inference_model(
    model,
    loader,
    model_name: str,
    device,
    id2label: dict | None = None,
) -> tuple[list[int], list[list[float]], list[str], list[int]]:
    """
    Run inference without ground-truth labels.

    Returns
    -------
    pred_label_ids   : list[int]   — predicted class indices
    pred_probs       : list[list[float]] — full softmax vector per sample
    pred_label_names : list[str]   — human-readable class names (or str(id) if id2label missing)
    pred_indices     : list[int]   — original dataframe indices
    """
    model.eval()
    pred_label_ids, pred_probs, pred_indices = [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = run_model_pred(model, batch, model_name)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            pred_label_ids.extend(preds.cpu().tolist())
            pred_probs.extend(probs.cpu().tolist())
            if "index" in batch:
                pred_indices.extend(batch["index"].cpu().tolist())
            else:
                pred_indices.extend(range(
                    len(pred_label_ids) - len(preds), len(pred_label_ids)
                ))

    pred_label_names = [
        id2label.get(i, str(i)) for i in pred_label_ids
    ] if id2label else [str(i) for i in pred_label_ids]

    print(f"Inference done — {len(pred_label_ids)} predictions")
    print("Predicted distribution:", Counter(pred_label_names))
    return pred_label_ids, pred_probs, pred_label_names, pred_indices


# ═══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ═══════════════════════════════════════════════════════════════════════════════

def train(
    args,
    model,
    pretrained_model: str,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    rank: int = 0,
    id2label: dict | None = None,
):
    """
    Full DDP training loop for multi-class target classification.

    Key differences vs binary version
    ───────────────────────────────────
    • No threshold sweep — predictions are always argmax.
    • Validation tracks macro F1 (best model selection).
    • Logs macro_f1, weighted_f1, per_class_f1 to wandb.
    • classification_report printed after every epoch on rank 0.
    """
    num_epochs = args.epochs
    model_name = args.model_name
    validation = args.validation
    size = args.size

    distributed = isinstance(
        train_loader.sampler, torch.utils.data.distributed.DistributedSampler
    )

    total_train_batches = torch.tensor(len(train_loader), device=device)
    dist.all_reduce(total_train_batches, op=dist.ReduceOp.SUM)

    if rank == 0:
        print(f"Total training batches (all GPUs): {total_train_batches.item()}")
        print(f"Training batches (this rank):      {len(train_loader)}")
        if validation and val_loader:
            print(f"Validation batches:                {len(val_loader)}")
        print(f"Test batches:                      {len(test_loader)}")
        print(f"Pretrained encoder: {pretrained_model} | Model: {model_name} | "
              f"Epochs: {num_epochs} | Size: {size}")

    best_val_macro_f1 = float("-inf")
    best_model = model
    patience = 10
    trigger_times = 0
    grad_accum_steps = 8

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_check_path = (
        f"models/checkpoints/{timestamp}_{pretrained_model.replace('/', '_')}_{size}_target.pt"
    )
    if rank == 0:
        os.makedirs(os.path.dirname(model_check_path), exist_ok=True)
        print(f"Checkpoint path: {model_check_path}")

    target_names = [id2label[i] for i in sorted(id2label)] if id2label else None

    # ── Epoch loop ─────────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        running_loss = 0.0
        running_corrects = 0
        true_labels, pred_labels = [], []
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for i, batch in enumerate(progress_bar):
            with autocast():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = run_model_pred(model, batch, model_name)
                labels = batch["labels"]
                logits = outputs.logits
                loss = criterion(logits, labels) / grad_accum_steps
                loss.backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # unscale before logging
            unscaled_loss = loss.detach().item() * grad_accum_steps
            running_loss += unscaled_loss
            preds = _predictions_from_logits(logits.detach())
            running_corrects += (preds == labels).sum().item()
            true_labels.extend(labels.cpu().tolist())
            pred_labels.extend(preds.cpu().tolist())
            progress_bar.set_postfix(loss=f"{unscaled_loss:.4f}")

        # ── Gather across GPUs ─────────────────────────────────────────────────
        gathered_true = [None] * dist.get_world_size()
        gathered_pred = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_true, true_labels)
        dist.all_gather_object(gathered_pred, pred_labels)
        all_true = [l for sub in gathered_true for l in sub]
        all_pred = [p for sub in gathered_pred for p in sub]

        total_loss     = torch.tensor(running_loss,     device=device)
        total_corrects = torch.tensor(running_corrects, device=device)
        total_samples  = torch.tensor(len(true_labels), device=device)
        dist.all_reduce(total_loss,     op=dist.ReduceOp.SUM)
        dist.all_reduce(total_corrects, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples,  op=dist.ReduceOp.SUM)

        avg_loss    = total_loss.item()     / total_samples.item()
        epoch_acc   = total_corrects.item() / total_samples.item()

        if rank == 0:
            train_metrics = _compute_metrics(all_true, all_pred, id2label)

            # Per-class F1 dict for wandb
            per_class_log = {
                f"train_f1_{target_names[i] if target_names else i}": v
                for i, v in enumerate(train_metrics["per_class_f1"])
            }
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": epoch_acc,
                "train_macro_f1": train_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "train_macro_precision": train_metrics["macro_precision"],
                "train_macro_recall": train_metrics["macro_recall"],
                **per_class_log,
            })
            print(
                f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | "
                f"Acc: {epoch_acc:.4f} | Macro-F1: {train_metrics['macro_f1']:.4f} | "
                f"Weighted-F1: {train_metrics['weighted_f1']:.4f}"
            )

        # ── Validation ────────────────────────────────────────────────────────
        if validation and val_loader and rank == 0:
            val_metrics = evaluate_model(
                model, val_loader, model_name, device,
                output_file=f"{model_name}_{size}_epoch{epoch+1}_val.json",
                criterion=criterion,
                id2label=id2label,
            )

            per_class_log = {
                f"val_f1_{target_names[i] if target_names else i}": v
                for i, v in enumerate(val_metrics["per_class_f1"])
            }
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "val_macro_precision": val_metrics["macro_precision"],
                "val_macro_recall": val_metrics["macro_recall"],
                **per_class_log,
            })
            print(
                f"Validation | Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Weighted-F1: {val_metrics['weighted_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_val_macro_f1:
                print(
                    f"  ↑ Best val Macro-F1: "
                    f"{best_val_macro_f1:.4f} → {val_metrics['macro_f1']:.4f}"
                )
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_model = model
                trigger_times = 0
                torch.save(model.state_dict(), model_check_path)
            else:
                trigger_times += 1
                if trigger_times > patience:
                    print("Early stopping!")
                    break

    if not validation and rank == 0:
        torch.save(model.state_dict(), model_check_path)

    # ── Final test evaluation ──────────────────────────────────────────────────
    if rank == 0:
        print("\nRunning final test evaluation ...")
        test_metrics = evaluate_model(
            best_model, test_loader, model_name, device,
            output_file=f"{model_name}_{size}_test.json",
            criterion=criterion,
            id2label=id2label,
        )

        per_class_log = {
            f"test_f1_{target_names[i] if target_names else i}": v
            for i, v in enumerate(test_metrics["per_class_f1"])
        }
        wandb.log({
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_weighted_f1": test_metrics["weighted_f1"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            **per_class_log,
        })
        print(
            f"Test | Loss: {test_metrics['loss']:.4f} | "
            f"Acc: {test_metrics['accuracy']:.4f} | "
            f"Macro-F1: {test_metrics['macro_f1']:.4f} | "
            f"Weighted-F1: {test_metrics['weighted_f1']:.4f}"
        )

    wandb.finish()