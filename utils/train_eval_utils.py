import torch
import pandas as pd 
import os
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer, LongformerTokenizer
from models.model import all_model_names
import wandb
from tqdm import tqdm 
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler
from utils.construct_graph import get_graph, get_hetero_graph
import json
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


tokenizerRobertaHS = AutoTokenizer.from_pretrained("camembert-base")

'''
def precision_score(true_labels, predicted_labels):
    true_positive = 0
    false_positive = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if predicted_label == 1:
            if true_label == predicted_label:
                true_positive += 1
            else:
                false_positive += 1
    if true_positive + false_positive == 0:
        return 0.0  # Avoid division by zero
    precision = true_positive / (true_positive + false_positive)
    return precision

def recall_score(true_labels, predicted_labels):
    true_positive = 0
    false_negative = 0

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == 1:
            if true_label == predicted_label:
                true_positive += 1
            else:
                false_negative += 1


    if true_positive + false_negative == 0:
        return 0.0  # Avoid division by zero

    recall = true_positive / (true_positive + false_negative)
    return recall

def f1_score(true_labels, predicted_labels):
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    if precision + recall == 0:
        return 0.0  # Avoid division by zero
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
'''

def get_criterion(device, balanced=False, class_counts=[]):
    if not balanced:
        class_counts = torch.tensor(class_counts) 
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum()
        class_weights = torch.tensor([0.3, 0.7]) # going for less extreme class weights.
        class_weights = class_weights.to(device)
        print('class weights ', class_weights, ' class 0 should have lower weight since it has more samples')
        
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        return nn.CrossEntropyLoss()

def get_context_texts(conv_array, index):
    my_id = conv_array[index][0]['id']
    parent_id = conv_array[index][0]['parent_id']

    parent_id, parent_text, post_id, post_text = "", "", "", ""
    for i, comment in enumerate(conv_array):
        if parent_id == comment[0]['id']:
            parent_text = comment[0]['body']
        if not parent_id or parent_id == "" or parent_id == "NA":
            post_id = comment[0]['id']
            post_text = comment[0]['body']
    if post_id == parent_id:
        return [parent_text]
    return [post_text, parent_text]

def get_context_texts_all(conv_array, index, conv_indices_to_keep=[]):
    my_id = conv_array[index][0]['id']
    all_texts = []
    for i, comment in enumerate(conv_array):
        if my_id != comment[0]['id'] and (len(conv_indices_to_keep) == 0 or i in conv_indices_to_keep):
            all_texts.append(comment[0]['body'])
    return all_texts

def get_all_texts(conv_array):
    all_texts = []
    for comment in conv_array:
        all_texts.append(comment[0]['body'])
    return all_texts


def get_post_parent(conv_array, index):
    my_id = conv_array[index][0]['id']
    parent_id = conv_array[index][0]['parent_id']

    parent_id, parent_text, post_id, post_text = "", "", "", ""
    for i, comment in enumerate(conv_array):
        if parent_id == comment[0]['id']:
            parent_text = comment[0]['body']
        if not parent_id or parent_id == "" or parent_id == "NA":
            post_id = comment[0]['id']
            post_text = comment[0]['body']

    return [post_text, parent_text]

def get_reactions_texts(conv_array, index):
  # we keep the target node, and all the nodes belonging from the tree having target_node as its root.
    target_comment = conv_array[index]
    target_timestamp = target_comment[0]['created_utc']
    target_id = target_comment[0]['id']

    parent_ids_to_keep = [target_id]
    reaction_txts = []

    for _, node in enumerate(conv_array):
      # we only want comments posted after or at the target node posted time
      if node[0]['created_utc'] >= target_timestamp:
        node_id = node[0]['id']
        parent_id = node[0]['parent_id']

        # we only want comments that root back to the target node, without including the target node
        if parent_id in parent_ids_to_keep and node_id != target_id:
          parent_ids_to_keep += [node_id]
          reaction_txts += [node[0]['body']]
    return reaction_txts

# Define validation function
def evaluate_model(model, loader, model_name, device, output_file="", tune_threshold=True, best_threshold=0.5):
    model.eval()
    thresholds = np.arange(0.1, 1.0, 0.1) if tune_threshold else [best_threshold]
    best_f1 = 0.0
    selected_threshold = best_threshold

    with torch.no_grad():
        with open(output_file, 'w') if output_file else None as outfile:
            for t in thresholds:
                running_loss = 0.0
                running_corrects = 0
                true_labels = []
                predicted_labels = []
                for batch in loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = run_model_pred(model, batch, model_name)
                        labels = batch["labels"]
                        loss = outputs.loss
                        running_loss, running_corrects, true_labels, predicted_labels = update_running_metrics(
                            loss, outputs, labels, running_loss, running_corrects, true_labels, predicted_labels, threshold=t
                            )
                f1 = f1_score(true_labels, predicted_labels, zero_division=0)
                if tune_threshold and f1 > best_f1:
                    best_f1 = f1
                    selected_threshold = t
                    print(f"Best threshold: {selected_threshold}, F1: {best_f1}")
                elif not tune_threshold:
                    best_f1 = f1  
                

        with open(output_file, 'w') if output_file else None as outfile:
            running_loss = 0.0
            running_corrects = 0
            true_labels = []
            predicted_labels = []
            for batch in loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = run_model_pred(model, batch, model_name)
                    labels = batch["labels"]
                    loss = outputs.loss
                    running_loss, running_corrects, true_labels, predicted_labels = update_running_metrics(
                        loss, outputs, labels, running_loss, running_corrects, true_labels, predicted_labels, threshold=best_threshold
                        )
            if outfile:
                preds = torch.argmax(outputs.logits, dim=1)
                my_output = {'pred_labels' : preds.detach().cpu().numpy().tolist(), 'y': labels.detach().cpu().numpy().tolist(), 'y_pred': outputs.logits.detach().cpu().numpy().tolist()}
                if model_name == "text-class":
                    text_indices =  batch["index"].detach().cpu().numpy().tolist()
                    my_output['index'] = text_indices
                json.dump(my_output, outfile)
                outfile.write('\n')
                    
                                        
    # Calculate average loss
    num_samples = len(true_labels)
    avg_loss = running_loss / num_samples
    accuracy = float(running_corrects) / num_samples
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    return avg_loss, accuracy, f1, precision, recall, selected_threshold

def run_model_pred(model, batch, model_name):
    if model_name == "text-class":
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
            labels=batch.get("labels", None),
        )
    return outputs


def update_running_metrics(loss, outputs, labels, running_loss, running_corrects, true_labels, predicted_labels, threshold=0.5):
    # Detach loss and add to running loss
    running_loss += loss.item()

    # Get predicted class (argmax over logits)
    #preds = torch.argmax(outputs.logits, dim=1)
    probs = torch.softmax(outputs.logits, dim=1)
    preds = (probs[:, 1] > threshold).long()

    # Count correct predictions
    running_corrects += torch.sum(preds == labels).item()

    # Store labels for metrics
    true_labels.extend(labels.detach().cpu().tolist())
    predicted_labels.extend(preds.detach().cpu().tolist())

    return running_loss, running_corrects, true_labels, predicted_labels


def train(args, model, pretrained_model, train_loader, val_loader, test_loader, criterion, optimizer, device, rank=0):
    num_epochs, model_name, validation, size = args.epochs, args.model_name, args.validation, args.size
    distributed = isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler)

    if rank == 0:
        print("Training set size: ", len(train_loader))
        print("Validation set size: ", len(val_loader))
        print("Test set size: ", len(test_loader))
        print("Train: epochs=", num_epochs, ", dataset_name=pointdarret", ", model=", model_name, "pretrained_model=", pretrained_model)

    best_val_f1 = float('-inf')
    best_model = model 
    # Patience is the maximum number of epoch with decaying validation scores we will wait for, before early stopping the training
    patience = 10
    trigger_times = 0

    # Generate a unique model name string to save the model at
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_check_path = f"models/checkpoints/{timestamp}_{pretrained_model}_{size}.pt"

    if rank == 0:
        os.makedirs(os.path.dirname(model_check_path), exist_ok=True)
        print(f"Saving model to ", model_check_path)

    best_threshold = 0.5

    # Training loop
    for epoch in range(num_epochs):
        if distributed:
            train_loader.sampler.set_epoch(epoch)  # required for shuffling in DDP

        running_loss = float(0)
        running_corrects = 0
        true_labels = []
        predicted_labels = []
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")


        for batch in progress_bar:
            with autocast():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = run_model_pred(model, batch, model_name)
                #loss = outputs.loss
                labels = batch["labels"]
                logits = outputs.logits
                loss = criterion(logits, labels) # use custom loss function

                # Backpropagate
                loss.backward()
                optimizer.step()
                #lr_scheduler.step()
                optimizer.zero_grad()
                
                # clear unused memory to reduce fragmentation
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

                running_loss, running_corrects, true_labels, predicted_labels = update_running_metrics(
                    loss, outputs, labels, running_loss, running_corrects, true_labels, predicted_labels, threshold=best_threshold
                    )
                progress_bar.set_postfix(loss=loss.item())

        if rank == 0:
            print("Unique predicted labels:", set(predicted_labels))
            print("True label distribution:", pd.Series(true_labels).value_counts())
            print("Predicted label distribution:", pd.Series(predicted_labels).value_counts())
            print("True label:", true_labels)
            print("Predicted label:", predicted_labels)

        num_samples = len(true_labels)
        avg_loss = running_loss/ num_samples
        epoch_accuracy = float(running_corrects) / num_samples
        epoch_precision = precision_score(true_labels, predicted_labels, zero_division=0)
        epoch_recall = recall_score(true_labels, predicted_labels, zero_division=0)
        epoch_f1 = f1_score(true_labels, predicted_labels, zero_division=0)

        if rank == 0:
            print(classification_report(true_labels, predicted_labels, digits=4))
            print(confusion_matrix(true_labels, predicted_labels))
            

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "train_accuracy": epoch_accuracy,
                "train_precision": epoch_precision,
                "train_recall": epoch_recall,
                "train_f1": epoch_f1
            })
        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_loss:.4f}, "
            f"Train Accuracy: {epoch_accuracy:.4f}, Train Precision: {epoch_precision:.4f}, "
            f"Train Recall: {epoch_recall:.4f}, Train F1 Score: {epoch_f1:.4f}")

        # If validation, compute and report the main metrics on the validation set
        if validation and rank == 0:
            avg_val_loss, val_accuracy, val_f1, val_precision, val_recall, val_best_threshold = evaluate_model(model, val_loader, model_name, device, f"{model_name}_{size}_{epoch}_val_outputs.tsv")
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1
            })
            
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
                f"Val Recall: {val_recall:.4f}, Val F1 Score: {val_f1:.4f}")
            
            # Update best validation f1 score, best model and save checkpoint
            if val_f1 > best_val_f1:
                print("Replacing best validation F1 score from ", best_val_f1 , " to ", val_f1, " best threshold from ", best_threshold, " to ", val_best_threshold)
                best_val_f1 = val_f1
                best_model = model
                best_threshold = val_best_threshold
                trigger_times = 0
                torch.save(model.state_dict(), model_check_path)
            # Early stopping logic 
            else:
                trigger_times += 1
                if trigger_times > patience:
                    print('Early stopping!')
                    break
    if not validation and rank == 0:
        torch.save(model.state_dict(), model_check_path)

    # Finally, evaluate on the test set and report all metrics
    if rank == 0:
        print("Running evaluation ...")
        test_loss, test_accuracy, test_f1, test_precision, test_recall, selected_threshold = evaluate_model(best_model, test_loader, model_name, device, f"{model_name}_{size}_test_outputs.tsv", best_threshold=best_threshold)
        assert selected_threshold == best_threshold

        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })
        print(f"Test Loss: {test_loss:.4f}, "
            f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, "
            f"Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}, Threshold: {best_threshold:.4f}")

    # Finish the run
    wandb.finish()