import argparse
import json
import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed,
    EarlyStoppingCallback
)

import torch
from torch import nn
import wandb
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Train Camembert v2 text classifier")
    parser.add_argument("--data_path", type=str, default="data/clean_annotated_comments.csv")
    parser.add_argument("--output_dir", type=str, default="./camembertv2_results")
    parser.add_argument("--model_name", type=str, default="almanach/camembertv2-base")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    set_seed(args.seed)

    # Init wandb
    wandb.init(project="camembertv2-text-classification", config=vars(args))

    # Load the dataset 
    print("Loading dataset...")
    df = pd.read_csv(args.data_path, low_memory=False)
    if "stop" not in df.columns:
        raise ValueError("Expected 'stop' column in the CSV.")
    df['labels'] = df['stop'].apply(lambda x: int(0) if x == 'no_stop' else int(1))

    # Reduce the dataset size for testing (remove after)
    df = df.sample(n=100, random_state=42)

    # Encode labels 
    #label_encoder = LabelEncoder()
    #df['label_encoded'] = label_encoder.fit_transform(df['label'])
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df[['text', 'labels']])
    dataset = dataset.train_test_split(test_size=0.2)
    #dataset = dataset.rename_column('label_encoded', 'labels')
    num_labels = 2
    #print(f"Number of labels: {num_labels}")

    # Define a tokenizer and tokenize 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    print(f"Tokenized datasets: {tokenized_datasets} ")
    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Define the model 
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)


    training_args = TrainingArguments(
        output_dir="./camembertv2_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    """


    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    num_labels = len(label_encoder.classes_)

    print("Computing class weights...")
    class_counts = train_df["label_encoded"].value_counts().to_dict()
    total_samples = len(train_df)
    class_weights = {cls: total_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    print(f"Class weights {class_weights}")
    weights = [class_weights[cls] for cls in sorted(class_weights.keys())]
    class_weights_tensor = torch.tensor(weights).to(torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=100)
        tokenized["labels"] = examples["labels"]
        return tokenized


    tokenized_train = train_dataset.map(tokenize_function, batched=True, batch_size=32, num_proc=1)
    tokenized_test = test_dataset.map(tokenize_function, batched=True, batch_size=32, num_proc=1)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    class WeightedLossTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.model.device)
            self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
            "precision": precision_score(labels, preds, average="weighted"),
            "recall": recall_score(labels, preds, average="weighted"),
        }
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_test,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )
    trainer = WeightedLossTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    """

    print("Starting training...")
    trainer.train()

    print("Evaluating model...")
    metrics = trainer.evaluate()
    print(metrics)

    print("Saving model and tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Saving training arguments...")
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print("Done!")

if __name__ == "__main__":
    main()
