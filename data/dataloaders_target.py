from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np


DATA_SPLIT_SEED = 42

# Target categories for moderation classification
TARGET_CLASSES = [
    "contenu (crédibilité)",
    "contenu (manque d'intérêt)",
    "format",
    "utilisateurs",
    "source",
    "page/groupe",
]
# Build label2id and id2label mappings
LABEL2ID = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_LABELS = len(TARGET_CLASSES)

# Define context segment limits
LIMITS = {
    "text": 600,
    "title": 200,
    "url_title": 200,
    "description": 200,
    "parent_comment": 300,
    "account": 50,
    "domain": 50,
}


def truncate_text(text, limit):
    """Truncate text by character length safely."""
    if not text or not isinstance(text, str):
        return ""
    return text.strip()[:limit]


def normalize_target(value):
    """
    Normalize raw target values to canonical label names.
    Handles the 'utilisateurs' / 'utilisateur' merge and strips whitespace.
    Returns None if the value is not in TARGET_CLASSES.
    """
    if not value or not isinstance(value, str):
        return None
    value = value.strip().lower()
    # Merge singular and plural forms for 'utilisateurs'
    if value in ("utilisateurs", "utilisateur"):
        return "utilisateurs"
    # Try exact match (case-insensitive)
    for canonical in TARGET_CLASSES:
        if value == canonical.lower():
            return canonical
    return None


def get_dataloads(
    data_path,
    size="small",
    validation=False,
    test_size=0.2,
    tokenizer_name="camembert-base",
    batch_size=8,
    seed=DATA_SPLIT_SEED,
    distributed=False,
    rank=None,
    world_size=None,
    model_name="text_only",
    inference_only=False,
    tokenize=True,
    balance_strategy="oversample",  # "oversample", "undersample", or "none"
):
    """
    Load and prepare data for multi-class target classification.

    Parameters
    ----------
    data_path : str
        Path to the CSV file.
    size : str
        "small" (100 rows), "medium" (1000 rows), or "large" (all rows).
    validation : bool
        Whether to create a validation split.
    test_size : float
        Fraction of data to use as test set.
    tokenizer_name : str
        HuggingFace tokenizer name.
    batch_size : int
        DataLoader batch size.
    seed : int
        Random seed.
    distributed : bool
        Whether to use DistributedSampler.
    rank : int or None
        Process rank for distributed training.
    world_size : int or None
        Total number of processes for distributed training.
    model_name : str
        Feature concatenation strategy (same options as binary dataloader).
    inference_only : bool
        If True, labels are set to -1 (no ground truth required).
    tokenize : bool
        Whether to tokenize the data.
    balance_strategy : str
        How to handle class imbalance on the *training* set:
        - "oversample"  : upsample minority classes to match the majority.
        - "undersample" : downsample majority classes to match the minority.
        - "none"        : leave the training distribution as-is.

    Returns
    -------
    train_loader, test_loader, val_loader : DataLoader or None
    class_counts : list[int]  — per-class counts in the (balanced) training set
    class_weights : torch.Tensor — inverse-frequency weights for loss weighting
    label2id : dict
    id2label : dict
    """
    print(
        f"Loading dataset from {data_path}, size={size}, "
        f"validation={validation}, test_size={test_size}, "
        f"balance_strategy={balance_strategy} ..."
    )

    df = pd.read_csv(data_path, low_memory=False)

    # ------------------------------------------------------------------ #
    #  Required columns                                                    #
    # ------------------------------------------------------------------ #
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")
    df["text"] = df["text"].fillna("").astype(str)

    if inference_only:
        df["labels"] = -1
    else:
        if "target" not in df.columns:
            raise ValueError("CSV must contain a 'target' column.")
        df["target_clean"] = df["target"].apply(normalize_target)
        # Drop rows with unknown target values
        unknown = df["target_clean"].isna()
        if unknown.sum() > 0:
            print(
                f"  Warning: dropping {unknown.sum()} rows with unrecognised "
                f"target values: {df.loc[unknown, 'target'].unique().tolist()}"
            )
            df = df[~unknown].copy()
        df["labels"] = df["target_clean"].map(LABEL2ID).astype(int)

    # ------------------------------------------------------------------ #
    #  Optional context columns (same logic as binary dataloader)          #
    # ------------------------------------------------------------------ #
    optional_cols = {
        "_all": ["account_name", "parent_domain", "post_description",
                 "post_message", "post_title", "parent_text"],
        "parent": ["parent_text"],
        "article": ["post_description", "post_title"],
        "message": ["post_message"],
        "com": ["account_name"],
        "domain": ["parent_domain"],
    }
    for key, cols in optional_cols.items():
        if key in model_name:
            for col in cols:
                if col not in df.columns:
                    raise ValueError(
                        f"CSV must contain '{col}' column for model_name '{model_name}'."
                    )
                df[col] = df[col].fillna("").astype(str)

    # ------------------------------------------------------------------ #
    #  Index                                                               #
    # ------------------------------------------------------------------ #
    df["index"] = range(len(df))

    # ------------------------------------------------------------------ #
    #  Size sampling                                                       #
    # ------------------------------------------------------------------ #
    if size == "small":
        df = df.sample(n=min(100, len(df)), random_state=seed)
    elif size == "medium":
        df = df.sample(n=min(1000, len(df)), random_state=seed)
    elif size == "large":
        pass
    else:
        raise ValueError("size must be one of: small, medium, large")

    # ------------------------------------------------------------------ #
    #  Train / test (/ val) split — stratified on labels                  #
    # ------------------------------------------------------------------ #
    if inference_only and test_size == 1.0:
        test_df = df.copy()
        train_df = pd.DataFrame(columns=df.columns)
        val_df = pd.DataFrame(columns=df.columns)
    else:
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["labels"], random_state=seed
        )
        val_df = pd.DataFrame(columns=df.columns)
        if validation:
            train_df, val_df = train_test_split(
                train_df,
                test_size=0.2,
                stratify=train_df["labels"],
                random_state=seed,
            )

    # ------------------------------------------------------------------ #
    #  Class balancing on the training set                                 #
    # ------------------------------------------------------------------ #
    if not train_df.empty and balance_strategy != "none" and not inference_only:
        train_df = _balance_dataframe(train_df, strategy=balance_strategy, seed=seed)

    # ------------------------------------------------------------------ #
    #  Class weights (for weighted cross-entropy loss)                     #
    # ------------------------------------------------------------------ #
    class_weights = torch.ones(NUM_LABELS)
    class_counts = []
    if not train_df.empty and not inference_only:
        counts = train_df["labels"].value_counts(sort=False)
        class_counts = [counts.get(i, 0) for i in range(NUM_LABELS)]
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.arange(NUM_LABELS),
            y=train_df["labels"].values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float)

    # ------------------------------------------------------------------ #
    #  Tokenizer & tokenisation functions                                  #
    # ------------------------------------------------------------------ #
    tokenizer = None
    if tokenize and tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # --- column selection per model_name ---
    cols, tok_func = _get_cols_and_tok_func(model_name, tokenizer)

    # ------------------------------------------------------------------ #
    #  Build HuggingFace Datasets                                          #
    # ------------------------------------------------------------------ #
    train_dataset, test_dataset, val_dataset = None, None, None

    if not train_df.empty:
        train_dataset = Dataset.from_pandas(train_df[cols].reset_index(drop=True))
        if tokenize and tok_func:
            train_dataset = train_dataset.map(tok_func, batched=True)

    if not test_df.empty:
        test_dataset = Dataset.from_pandas(test_df[cols].reset_index(drop=True))
        if tokenize and tok_func:
            test_dataset = test_dataset.map(tok_func, batched=True)

    if validation and not val_df.empty:
        val_dataset = Dataset.from_pandas(val_df[cols].reset_index(drop=True))
        if tokenize and tok_func:
            val_dataset = val_dataset.map(tok_func, batched=True)

    # ------------------------------------------------------------------ #
    #  Set torch format & collator                                         #
    # ------------------------------------------------------------------ #
    if tokenize:
        if "_embed" in model_name and model_name != "context_all_embed":
            columns = [
                "text_input_ids", "text_attention_mask",
                "ctx_input_ids", "ctx_attention_mask",
                "labels", "index",
            ]
            data_collator = _collate_dual
        elif model_name == "context_all_embed":
            columns = [
                "text_input_ids", "text_attention_mask",
                "post_input_ids", "post_attention_mask",
                "url_input_ids", "url_attention_mask",
                "parent_input_ids", "parent_attention_mask",
                "labels", "index",
            ]
            data_collator = _collate_four
        else:
            columns = ["input_ids", "attention_mask", "labels", "index"]
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        for ds in (train_dataset, test_dataset, val_dataset):
            if ds is not None:
                ds.set_format("torch", columns=columns)
    else:
        data_collator = _collate_raw

    # ------------------------------------------------------------------ #
    #  DataLoaders                                                         #
    # ------------------------------------------------------------------ #
    train_loader, test_loader, val_loader = None, None, None

    if train_dataset:
        sampler = (
            DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            if distributed else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=data_collator,
        )

    if test_dataset:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )

    if val_dataset:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator
        )

    return train_loader, test_loader, val_loader, class_counts, class_weights, LABEL2ID, ID2LABEL


# ====================================================================== #
#  Helpers                                                                 #
# ====================================================================== #

def _balance_dataframe(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    """Return a resampled copy of df with balanced class distribution."""
    counts = df["labels"].value_counts()
    if strategy == "oversample":
        target_n = counts.max()
        parts = [
            df[df["labels"] == label].sample(n=target_n, replace=True, random_state=seed)
            for label in counts.index
        ]
    elif strategy == "undersample":
        target_n = counts.min()
        parts = [
            df[df["labels"] == label].sample(n=target_n, replace=False, random_state=seed)
            for label in counts.index
        ]
    else:
        return df
    balanced = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  Balanced training set ({strategy}): {len(balanced)} rows, "
          f"{target_n} per class.")
    return balanced


def _collate_raw(batch):
    return {
        "text": [x["text"] for x in batch],
        "labels": torch.tensor([x["labels"] for x in batch]),
        "index": torch.tensor([x["index"] for x in batch]),
    }


def _collate_dual(batch):
    return {
        "text_input_ids": torch.stack([x["text_input_ids"] for x in batch]),
        "text_attention_mask": torch.stack([x["text_attention_mask"] for x in batch]),
        "ctx_input_ids": torch.stack([x["ctx_input_ids"] for x in batch]),
        "ctx_attention_mask": torch.stack([x["ctx_attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "index": torch.stack([x["index"] for x in batch]),
    }


def _collate_four(batch):
    return {
        "text_input_ids": torch.stack([x["text_input_ids"] for x in batch]),
        "text_attention_mask": torch.stack([x["text_attention_mask"] for x in batch]),
        "post_input_ids": torch.stack([x["post_input_ids"] for x in batch]),
        "post_attention_mask": torch.stack([x["post_attention_mask"] for x in batch]),
        "url_input_ids": torch.stack([x["url_input_ids"] for x in batch]),
        "url_attention_mask": torch.stack([x["url_attention_mask"] for x in batch]),
        "parent_input_ids": torch.stack([x["parent_input_ids"] for x in batch]),
        "parent_attention_mask": torch.stack([x["parent_attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]),
        "index": torch.stack([x["index"] for x in batch]),
    }


def _get_cols_and_tok_func(model_name, tokenizer):
    """
    Return (column list, tokenisation function) for a given model_name.
    tokenizer may be None if tokenize=False — in that case tok_func is None too.
    """
    sep = (tokenizer.sep_token or "[SEP]") if tokenizer else "[SEP]"

    # ---- tokenisation closures ----

    def tokenize_text_only(batch):
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_article_concat(batch):
        texts = [
            f"{t} {sep} {title} {sep} {desc}"
            for t, title, desc in zip(batch["text"], batch["post_title"], batch["post_description"])
        ]
        tokens = tokenizer(texts, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_message_concat(batch):
        texts = [f"{t} {sep} {m}" for t, m in zip(batch["text"], batch["post_message"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_parent_concat(batch):
        texts = [f"{t} {sep} {p}" for t, p in zip(batch["text"], batch["parent_text"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_commu_concat(batch):
        texts = [f"{acc} {sep} {t}" for acc, t in zip(batch["account_name"], batch["text"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_domain_concat(batch):
        texts = [f"{dom} {sep} {t}" for dom, t in zip(batch["parent_domain"], batch["text"])]
        tokens = tokenizer(texts, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_all_concat(batch):
        concat_texts = []
        for text, post_msg, title, desc, parent, account, domain in zip(
            batch["text"], batch["post_message"], batch["post_title"],
            batch["post_description"], batch["parent_text"],
            batch["account_name"], batch["parent_domain"],
        ):
            text = truncate_text(text, LIMITS["text"])
            post_msg = truncate_text(post_msg, LIMITS["title"])
            title = truncate_text(title, LIMITS["url_title"])
            desc = truncate_text(desc, LIMITS["description"])
            parent = truncate_text(parent, LIMITS["parent_comment"])
            account = truncate_text(account, LIMITS["account"])
            domain = truncate_text(domain, LIMITS["domain"])
            sections = [text]
            if parent:
                sections.append(f"[PARENT] {parent}")
            if account:
                sections.append(f"[ACCOUNT] {account}")
            if post_msg:
                sections.append(f"[POST] {post_msg}")
            if domain:
                sections.append(f"[DOMAIN] {domain}")
            if title or desc:
                sections.append(f"[ARTICLE] {title} {desc}".strip())
            concat_texts.append(f" {sep} ".join(sections).strip())
        tokens = tokenizer(concat_texts, truncation=True, padding="max_length", max_length=512)
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_dual(batch):
        output = {}
        ctx_text = batch["text"]
        if all(k in batch for k in ["parent_domain", "account_name", "post_title",
                                     "post_description", "post_message", "parent_text"]):
            concat_texts = []
            for post_msg, title, desc, parent, account, domain in zip(
                batch["post_message"], batch["post_title"], batch["post_description"],
                batch["parent_text"], batch["account_name"], batch["parent_domain"],
            ):
                sections = []
                if truncate_text(parent, LIMITS["parent_comment"]):
                    sections.append(f"[PARENT] {truncate_text(parent, LIMITS['parent_comment'])}")
                if truncate_text(account, LIMITS["account"]):
                    sections.append(f"[ACCOUNT] {truncate_text(account, LIMITS['account'])}")
                if truncate_text(post_msg, LIMITS["title"]):
                    sections.append(f"[POST] {truncate_text(post_msg, LIMITS['title'])}")
                if truncate_text(domain, LIMITS["domain"]):
                    sections.append(f"[DOMAIN] {truncate_text(domain, LIMITS['domain'])}")
                t = truncate_text(title, LIMITS["url_title"])
                d = truncate_text(desc, LIMITS["description"])
                if t or d:
                    sections.append(f"[ARTICLE] {t} {d}".strip())
                concat_texts.append(f" {sep} ".join(sections).strip())
            ctx_text = concat_texts
        elif "parent_domain" in batch:
            ctx_text = batch["parent_domain"]
        elif "account_name" in batch:
            ctx_text = batch["account_name"]
        elif "post_title" in batch and "post_description" in batch:
            ctx_text = [f"{t} {sep} {d}" for t, d in zip(batch["post_title"], batch["post_description"])]
        elif "post_message" in batch:
            ctx_text = batch["post_message"]
        elif "parent_text" in batch:
            ctx_text = batch["parent_text"]
        text_tok = tokenizer(batch["text"], truncation=True, padding="max_length")
        ctx_tok = tokenizer(ctx_text, truncation=True, padding="max_length")
        for k in text_tok:
            output[f"text_{k}"] = text_tok[k]
        for k in ctx_tok:
            output[f"ctx_{k}"] = ctx_tok[k]
        output["labels"] = batch["labels"]
        output["index"] = batch["index"]
        return output

    def tokenize_four(batch):
        output = {}
        text_tok = tokenizer(batch["text"], truncation=True, padding="max_length")
        post_strings = [f"{a}[SEP]{b}" for a, b in zip(batch["account_name"], batch["post_message"])]
        post_tok = tokenizer(post_strings, truncation=True, padding="max_length")
        url_strings = [f"{d}[SEP]{t}{desc}" for d, t, desc in zip(batch["parent_domain"], batch["post_title"], batch["post_description"])]
        url_tok = tokenizer(url_strings, truncation=True, padding="max_length")
        parent_tok = tokenizer(batch["parent_text"], truncation=True, padding="max_length")
        for k in text_tok:
            output[f"text_{k}"] = text_tok[k]
        for k in post_tok:
            output[f"post_{k}"] = post_tok[k]
        for k in url_tok:
            output[f"url_{k}"] = url_tok[k]
        for k in parent_tok:
            output[f"parent_{k}"] = parent_tok[k]
        output["labels"] = batch["labels"]
        output["index"] = batch["index"]
        return output

    # ---- routing ----
    all_ctx_cols = [
        "text", "parent_domain", "account_name", "post_title",
        "post_description", "post_message", "parent_text", "labels", "index",
    ]
    mapping = {
        "text_only":              (["text", "labels", "index"],                              tokenize_text_only),
        "parent_text_concat":     (["text", "parent_text", "labels", "index"],               tokenize_parent_concat),
        "article_text_concat":    (["text", "post_title", "post_description", "labels", "index"], tokenize_article_concat),
        "message_text_concat":    (["text", "post_message", "labels", "index"],              tokenize_message_concat),
        "com_text_concat":        (["text", "account_name", "labels", "index"],              tokenize_commu_concat),
        "domain_text_concat":     (["text", "parent_domain", "labels", "index"],             tokenize_domain_concat),
        "context_all_text_embed": (all_ctx_cols,                                             tokenize_dual),
        "context_all_text_concat":(all_ctx_cols,                                             tokenize_all_concat),
        "context_all_embed":      (all_ctx_cols,                                             tokenize_four),
    }
    cols, tok_func = mapping.get(model_name, (["text", "labels", "index"], tokenize_text_only))
    if tokenizer is None:
        tok_func = None
    return cols, tok_func