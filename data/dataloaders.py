from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
import torch
import pandas as pd


def get_dataloads(
    data_path, size="small", validation=False, test_size=0.2,
    tokenizer_name="camembert-base", batch_size=8, seed=42,
    distributed=False, rank=None, world_size=None,
    model_name="text_only"  # options: text_only, post_text_concat, post_text_embed
):
    
    print(f"Loading dataset from {data_path}, size is {size}, validation is {validation}, test size is {test_size}...")
    
    df = pd.read_csv(data_path, low_memory=False)
    
    if "stop" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'stop' and 'text' columns.")
    if "account_name" not in df.columns:
        raise ValueError("CSV must contain 'account_name' column for community-based model_name options.")
    if "page_group_type" not in df.columns:
        raise ValueError("CSV must contain 'page_group_type' column for community-based model_name options.")
    if model_name in {"post_text_concat", "post_text_embed"} and "share_title" not in df.columns:
        raise ValueError("CSV must contain 'post_title' column for this model_name.")
    
    df['labels'] = df['stop'].apply(lambda x: 0 if x == 'no_stop' else 1)
    df['index'] = range(1, len(df) + 1)

    # clean string fields and fill Nan with empty string
    df["share_title"] = df["share_title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["account_name"] = df["account_name"].fillna("").astype(str)
    df["page_group_type"] = df["page_group_type"].fillna("").astype(str)

    df["parent_domain"] = df["parent_domain"].fillna("").astype(str)
    df["source_type"] = df["source_type"].fillna("").astype(str)
    df["theme"] = df["theme"].fillna("").astype(str)


    if size == "small":
        df = df.sample(n=100, random_state=seed)
    elif size == "medium":
        df = df.sample(n=1000, random_state=seed)
    elif size == "large":
        pass
    else:
        raise ValueError("Size must be one of: small, medium, large")

    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['labels'], random_state=seed)
    if validation:
        train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=seed)
    
    class_counts = train_df['labels'].value_counts(sort=False).tolist()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_text_only(batch):
        tokens = tokenizer(batch["text"], truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_post_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [t + f" {sep_token} " + title for t, title in zip(batch["text"], batch["share_title"])]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_commu_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{acc} {sep_token} {t}"
            for acc, t in zip(batch["account_name"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_commu_type_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{acc} {page_type} {sep_token} {t}"
            for acc, page_type, t in zip(batch["account_name"], batch["page_group_type"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens
    
    def tokenize_commu_post_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{acc} {sep_token} {t} {sep_token} {title}"
            for acc, t, title in zip(batch["account_name"], batch["text"], batch["share_title"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    def tokenize_commu_type_post_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{acc} {page_type} {sep_token} {t} {sep_token} {title}"
            for acc, page_type, t, title in zip(batch["account_name"], batch["page_group_type"], batch["text"], batch["share_title"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens

    # source
    def tokenize_domain_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{dom} {sep_token} {t}"
            for dom, t in zip(batch["parent_domain"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens    
    def tokenize_source_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{src} {sep_token} {t}"
            for src, t in zip(batch["source_type"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens
    def tokenize_theme_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{thm} {sep_token} {t}"
            for thm, t in zip(batch["theme"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens  
    def tokenize_domain_source_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{dom} {src} {sep_token} {t}"
            for dom, src, t in zip(batch["parent_domain"], batch["source_type"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens
    def tokenize_domain_theme_concat(batch):
        sep_token = tokenizer.sep_token or "[SEP]"
        concat_text = [
            f"{dom} {thm} {sep_token} {t}"
            for dom, thm, t in zip(batch["parent_domain"], batch["theme"], batch["text"])
        ]
        tokens = tokenizer(concat_text, truncation=True, padding="max_length")
        tokens["labels"] = batch["labels"]
        tokens["index"] = batch["index"]
        return tokens
    

    def tokenize_dual(batch):
        output = {}
        text_tok = tokenizer(batch["text"], truncation=True, padding="max_length")
        title_tok = tokenizer(batch["share_title"], truncation=True, padding="max_length")
        for k in text_tok:
            output[f"text_{k}"] = text_tok[k]
        for k in title_tok:
            output[f"title_{k}"] = title_tok[k]
        output["labels"] = batch["labels"]
        output["index"] = batch["index"]
        return output

    if model_name == "text_only":
        cols = ['text', 'labels', 'index']
        tok_func = tokenize_text_only
    elif model_name == "text_post_concat":
        cols = ['text', 'share_title', 'labels', 'index']
        tok_func = tokenize_post_concat
    elif model_name == "com_text_concat":
        cols = ['text', 'account_name', 'labels', 'index']
        tok_func = tokenize_commu_concat
    elif model_name == "com_text_post_concat":
        cols = ['text', 'account_name', 'share_title', 'labels', 'index']
        tok_func = tokenize_commu_post_concat
    elif model_name == "com_type_text_concat":
        cols = ['text', 'account_name', 'page_group_type', 'labels', 'index']
        tok_func = tokenize_commu_type_concat
    elif model_name == "com_type_text_post_concat":
        cols = ['text', 'account_name', 'page_group_type', 'share_title', 'labels', 'index']
        tok_func = tokenize_commu_type_post_concat
    elif model_name == "text_domain_concat":
        cols = ['text', 'parent_domain', 'labels', 'index']
        tok_func = tokenize_domain_concat
    elif model_name == "text_source_concat":
        cols = ['text', 'source_type', 'labels', 'index']
        tok_func = tokenize_source_concat
    elif model_name == "text_theme_concat":
        cols = ['text', 'theme', 'labels', 'index']
        tok_func = tokenize_theme_concat
    elif model_name == "text_domain_source_concat":
        cols = ['text', 'parent_domain', 'source_type', 'labels', 'index']
        tok_func = tokenize_domain_source_concat
    elif model_name == "text_domain_theme_concat":
        cols = ['text', 'parent_domain', 'theme', 'labels', 'index']
        tok_func = tokenize_domain_theme_concat
    else:  # post_text_embed
        cols = ['text', 'share_title', 'labels', 'index']
        tok_func = tokenize_dual



    #train_dataset = Dataset.from_pandas(train_df[cols]).map(tok_func, batched=True)
    #test_dataset = Dataset.from_pandas(test_df[cols]).map(tok_func, batched=True)
    #val_dataset = Dataset.from_pandas(val_df[cols]).map(tok_func, batched=True) if validation else None


    # Convert to HuggingFace Datasets
    train_dataset = Dataset.from_pandas(train_df[cols]).map(
        tok_func, batched=True
        )
    test_dataset = Dataset.from_pandas(test_df[cols]).map(
        tok_func, batched=True
    )
    if validation:
        val_dataset = Dataset.from_pandas(val_df[cols]).map(
            tok_func, batched=True
            )
    
    print("Train columns after mapping:", train_dataset.column_names)

    # Set output format
    if model_name == "text_only" or "_concat" in model_name:
        columns = ["input_ids", "attention_mask", "labels", "index"]
    else:  # post_text_embed
        columns = [
            "text_input_ids", "text_attention_mask",
            "title_input_ids", "title_attention_mask",
            "labels", "index"
        ]
    train_dataset.set_format("torch", columns=columns)
    test_dataset.set_format("torch", columns=columns)
    if val_dataset:
        val_dataset.set_format("torch", columns=columns)

    # Collator logic
    if model_name == "post_text_embed":
        def collate_dual(batch):
            return {
                'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                'title_input_ids': torch.stack([x['title_input_ids'] for x in batch]),
                'title_attention_mask': torch.stack([x['title_attention_mask'] for x in batch]),
                'labels': torch.stack([x['labels'] for x in batch]),
                'index': torch.stack([x['index'] for x in batch]),
            }
        data_collator = collate_dual
    else:
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=data_collator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator) if validation else None

    return train_loader, test_loader, val_loader, class_counts


def get_dataloads_old(
    data_path, size="small", validation=False, test_size=0.2,
    tokenizer_name="camembert-base", batch_size=8, seed=42,
    distributed=False, rank=None, world_size=None
):    
    print(f"Loading dataset from {data_path}, size is {size}, validation is {validation}, test size is {test_size}...")
    
    df = pd.read_csv(data_path, low_memory=False)
    
    if "stop" not in df.columns or "text" not in df.columns:
        raise ValueError("CSV must contain 'stop' and 'text' columns.")

    # Convert stop column to binary labels
    df['labels'] = df['stop'].apply(lambda x: 0 if x == 'no_stop' else 1)

    # Add 1-based index column
    df['index'] = range(1, len(df) + 1)
    
    df["account_name"] = df["account_name"].fillna("").astype(str)
    df["page_group_type"] = df["page_group_type"].fillna("").astype(str)

    # Sample the dataset by size
    if size == "small":
        df = df.sample(n=100, random_state=seed)
    elif size == "medium":
        df = df.sample(n=1000, random_state=seed)
    elif size == "large":
        pass  # use full dataset
    else:
        raise ValueError("Size must be one of: small, medium, large")

    # Split into train/test using stratified sampling
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['labels'], random_state=seed)


    if validation:
        # Further split train into train/val stratified
        train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=seed)
    
    class_counts = train_df['labels'].value_counts(sort=False).tolist()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")
    

    train_dataset = Dataset.from_pandas(train_df[['text', 'labels', 'index']]).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df[['text', 'labels', 'index']]).map(tokenize, batched=True)
    val_dataset = None
    if validation:
        val_dataset = Dataset.from_pandas(val_df[['text', 'labels', 'index']]).map(tokenize, batched=True)

    # Set PyTorch format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "index"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "index"])
    if val_dataset:
        val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "index"])

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Handle distributed cases
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=data_collator
    )

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator) if validation else None

    return train_loader, test_loader, val_loader, class_counts
