from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader, DistributedSampler


DATA_SPLIT_SEED = 42
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


def get_dataloads(
    data_path, size="small", validation=False, test_size=0.2,
    tokenizer_name="camembert-base", batch_size=8, seed=DATA_SPLIT_SEED,
    distributed=False, rank=None, world_size=None,
    model_name="text_only",  # options: text_only, post_text_concat, post_text_embed
    inference_only=False, tokenize=True
):
    
    print(f"Loading dataset from {data_path}, size is {size}, validation is {validation}, test size is {test_size}...")
    
    df = pd.read_csv(data_path, low_memory=False)
    
    if "text" not in df.columns:
        raise ValueError("CSV must contain 'text' columns.")
    df["text"] = df["text"].fillna("").astype(str)
    
    if inference_only: #inference_only:
        df['labels'] = -1
    else: # eval mode should have labels, i.e. 'stop' column   
        if "stop" not in df.columns:
            raise ValueError("CSV must contain 'stop' columns.")
        df['labels'] = df['stop'].apply(lambda x: 0 if x == 'no_stop' else 1)
    
    if "_all" in model_name:
        if "account_name" not in df.columns:
            raise ValueError("CSV must contain 'account_name' column for this model_name.")
        df["account_name"] = df["account_name"].fillna("").astype(str)        
        if "parent_domain" not in df.columns:
            raise ValueError("CSV must contain 'parent_domain' column for this model_name.")
        df["parent_domain"] = df["parent_domain"].fillna("").astype(str)
        if "post_description" not in df.columns:
            raise ValueError("CSV must contain 'post_description' column for this model_name.")
        df["post_description"] = df["post_description"].fillna("").astype(str)
        if "post_message" not in df.columns:
            raise ValueError("CSV must contain 'post_message' column for this model_name.")
        df["post_message"] = df["post_message"].fillna("").astype(str)
        if "post_title" not in df.columns:
            raise ValueError("CSV must contain 'post_title' column for this model_name.")
        df["post_title"] = df["post_title"].fillna("").astype(str)
        if "parent_text" not in df.columns:
            raise ValueError("CSV must contain 'parent_text' column for this model_name.")
        df["parent_text"] = df["parent_text"].fillna("").astype(str)
    if "parent" in model_name: #OK
        if "parent_domain" not in df.columns:
            raise ValueError("CSV must contain 'parent_domain' column for this model_name.")
        df["parent_text"] = df["parent_text"].fillna("").astype(str)
    if "article" in model_name: # OK 
        if "post_description" not in df.columns:
            raise ValueError("CSV must contain 'post_description' column for this model_name.")
        df["post_description"] = df["post_description"].fillna("").astype(str)
        if "post_title" not in df.columns:
            raise ValueError("CSV must contain 'post_title' column for this model_name.")
        df["post_title"] = df["post_title"].fillna("").astype(str)
    if "message" in model_name: #OK = post
        if "post_message" not in df.columns:
            raise ValueError("CSV must contain 'post_message' column for this model_name.")
        df["post_message"] = df["post_message"].fillna("").astype(str)
    if "com" in model_name: #OK
        if "account_name" not in df.columns:
            raise ValueError("CSV must contain 'account_name' column for this model_name.")
        df["account_name"] = df["account_name"].fillna("").astype(str)
    if "domain" in model_name: #OK
        if "parent_domain" not in df.columns:
            raise ValueError("CSV must contain 'parent_domain' column for this model_name.")
        df["parent_domain"] = df["parent_domain"].fillna("").astype(str)

    # create the index column, start with 0
    df["index"] = range(len(df))
    
    if size == "small":
        df = df.sample(n=100, random_state=seed)
    elif size == "medium":
        df = df.sample(n=1000, random_state=seed)
    elif size == "large":
        pass
    else:
        raise ValueError("Size must be one of: small, medium, large")
    
    if inference_only and test_size == 1.0:
        test_df = df.copy()
        train_df = pd.DataFrame(columns=df.columns)
        val_df = pd.DataFrame(columns=df.columns)
    else:
        train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['labels'], random_state=seed)
        if validation:
            train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['labels'], random_state=seed)

    train_dataset, test_dataset, val_dataset = None, None, None
    class_counts = []
    tokenizer = None
    if tokenize and tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        def tokenize_text_only(batch):
            tokens = tokenizer(batch["text"], truncation=True, padding="max_length")
            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]
            return tokens

        def tokenize_article_concat(batch):
            sep_token = tokenizer.sep_token or "[SEP]"
            concat_text = [t + f" {sep_token} " + title + f" {sep_token} " + desc for t, title, desc in zip(batch["text"], batch["post_title"], batch["post_description"])]
            tokens = tokenizer(concat_text, truncation=True, padding="max_length")
            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]
            return tokens
        
        def tokenize_message_concat(batch):
            sep_token = tokenizer.sep_token or "[SEP]"
            concat_text = [t + f" {sep_token} " + mesg for t, mesg in zip(batch["text"], batch["post_message"])]
            tokens = tokenizer(concat_text, truncation=True, padding="max_length")
            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]
            return tokens

        def tokenize_parent_concat(batch):
            sep_token = tokenizer.sep_token or "[SEP]"
            concat_text = [t + f" {sep_token} " + parent for t, parent in zip(batch["text"], batch["parent_text"])]
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
        
        def tokenize_all_concat(batch): # for ContextConcat
            sep = tokenizer.sep_token or "[SEP]"
            concat_texts = []
            #   ['text', 'parent_domain', 'account_name', 'post_title', 'post_description', 'post_message', 'parent_text', 'labels', 'index']

            for text, post_msg, title, desc, parent, account, domain in zip(
                batch["text"],
                batch["post_message"],
                batch["post_title"],
                batch["post_description"],
                batch["parent_text"],
                batch["account_name"],
                batch["parent_domain"],
            ):
                # Truncate fields
                text = truncate_text(text, LIMITS["text"])
                post_msg = truncate_text(post_msg, LIMITS["title"])
                title = truncate_text(title, LIMITS["url_title"])
                desc = truncate_text(desc, LIMITS["description"])
                parent = truncate_text(parent, LIMITS["parent_comment"])
                account = truncate_text(account, LIMITS["account"])
                domain = truncate_text(domain, LIMITS["domain"])

                # Build sections dynamically (only if not empty)
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
                    art = f"[ARTICLE] {title} {desc}".strip()
                    sections.append(art)

                # Join all with SEP token
                concat = f" {sep} ".join(sections)
                concat_texts.append(concat.strip())
            print(concat_texts)

            # Tokenize everything with truncation and padding
            tokens = tokenizer(
                concat_texts,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )

            tokens["labels"] = batch["labels"]
            tokens["index"] = batch["index"]

            return tokens
 
        def tokenize_dual(batch): # for ContextEmbed
            output = {}
            sep_token = tokenizer.sep_token or "[SEP]"
            ctx_text = batch["text"]
            if 'parent_domain' in batch.keys() and 'account_name' in batch.keys() and 'post_title' in batch.keys() and 'post_description' in batch.keys() and 'post_message' in batch.keys() and 'parent_text' in batch.keys():
                concat_texts = []
                #   ['text', 'parent_domain', 'account_name', 'post_title', 'post_description', 'post_message', 'parent_text', 'labels', 'index']
                for post_msg, title, desc, parent, account, domain in zip(
                    batch["post_message"],
                    batch["post_title"],
                    batch["post_description"],
                    batch["parent_text"],
                    batch["account_name"],
                    batch["parent_domain"],
                ):
                    # Truncate fields
                    post_msg = truncate_text(post_msg, LIMITS["title"])
                    title = truncate_text(title, LIMITS["url_title"])
                    desc = truncate_text(desc, LIMITS["description"])
                    parent = truncate_text(parent, LIMITS["parent_comment"])
                    account = truncate_text(account, LIMITS["account"])
                    domain = truncate_text(domain, LIMITS["domain"])

                    # Build sections dynamically (only if not empty)
                    sections = []
                    if parent:
                        sections.append(f"[PARENT] {parent}")
                    if account:
                        sections.append(f"[ACCOUNT] {account}")
                    if post_msg:
                        sections.append(f"[POST] {post_msg}")
                    if domain:
                        sections.append(f"[DOMAIN] {domain}")
                    if title or desc:
                        art = f"[ARTICLE] {title} {desc}".strip()
                        sections.append(art)

                    # Join all with SEP token
                    concat = f" {sep_token} ".join(sections)
                    concat_texts.append(concat.strip())
                ctx_text = concat_texts # will do the code after for all 
            elif 'parent_domain' in batch.keys():
                ctx_text = batch['parent_domain']
            elif 'account_name' in batch.keys():
                ctx_text = batch['account_name'] 
            elif 'post_title' in batch.keys() and 'post_description' in batch.keys():
                ctx_text = [title + f" {sep_token} " + desc for title, desc in zip(batch["post_title"], batch["post_description"])]
            elif 'post_message' in batch.keys():
                ctx_text = batch['post_message']
            elif 'parent_text' in batch.keys():
                ctx_text = batch['parent_text']
            
            text_tok = tokenizer(batch["text"], truncation=True, padding="max_length")
            ctx_tok = tokenizer(ctx_text, truncation=True, padding="max_length")
            for k in text_tok:
                output[f"text_{k}"] = text_tok[k]
            for k in ctx_tok:
                output[f"ctx_{k}"] = ctx_tok[k]
            output["labels"] = batch["labels"]
            output["index"] = batch["index"]
            print(output.keys())
            return output

        def tokenize_four(batch):
            output = {}

            # Tokenize text column
            text_tok = tokenizer(batch["text"], truncation=True, padding="max_length")

            # Concatenate post strings element-wise
            post_strings = [a + '[SEP]' + b for a, b in zip(batch["account_name"], batch["post_message"])]
            post_tok = tokenizer(post_strings, truncation=True, padding="max_length")

            # Concatenate url strings element-wise
            url_strings = [d + '[SEP]' + t + desc for d, t, desc in zip(batch["parent_domain"], batch["post_title"], batch["post_description"])]
            url_tok = tokenizer(url_strings, truncation=True, padding="max_length")

            # Tokenize parent text
            parent_tok = tokenizer(batch["parent_text"], truncation=True, padding="max_length")

            # Add tokenized outputs with prefixes
            for k in text_tok:
                output[f"text_{k}"] = text_tok[k]
            for k in post_tok:
                output[f"post_{k}"] = post_tok[k]
            for k in url_tok:
                output[f"url_{k}"] = url_tok[k]
            for k in parent_tok:
                output[f"parent_{k}"] = parent_tok[k]

            # Add labels and index
            output["labels"] = batch["labels"]
            output["index"] = batch["index"]

            return output


        if model_name == "text_only":
            cols = ['text', 'labels', 'index']
            tok_func = tokenize_text_only
        elif model_name == "parent_text_concat":
            cols = ['text', 'parent_text', 'labels', 'index']
            tok_func = tokenize_parent_concat
        elif model_name == "article_text_concat":
            cols = ['text', 'post_title', 'post_description', 'labels', 'index']
            tok_func = tokenize_article_concat
        elif model_name == "message_text_concat":
            cols = ['text', 'post_message', 'labels', 'index']
            tok_func = tokenize_message_concat
        elif model_name == "com_text_concat":
            cols = ['text', 'account_name', 'labels', 'index']
            tok_func = tokenize_commu_concat
        elif model_name == "domain_text_concat":
            cols = ['text', 'parent_domain', 'labels', 'index']
            tok_func = tokenize_domain_concat
        elif model_name == "parent_text_concat":
            cols = ['text', 'parent_text', 'labels', 'index']
            tok_func = tokenize_domain_concat
        # all context models
        elif model_name == "context_all_text_embed":
            cols = ['text', 'parent_domain', 'account_name', 'post_title', 'post_description', 'post_message', 'parent_text', 'labels', 'index']
            tok_func = tokenize_dual
        elif model_name == "context_all_text_concat":
            cols = ['text', 'parent_domain', 'account_name', 'post_title', 'post_description', 'post_message', 'parent_text', 'labels', 'index']
            tok_func = tokenize_all_concat
        elif model_name == "context_all_embed":
            cols = ['text', 'parent_domain', 'account_name', 'post_title', 'post_description', 'post_message', 'parent_text', 'labels', 'index']
            tok_func = tokenize_four
        else :
            cols = ['text', 'labels', 'index']

    # Convert to HuggingFace Datasets
    if not train_df.empty:
        train_dataset = Dataset.from_pandas(train_df[cols])
        if tokenize:
            train_dataset = train_dataset.map(tok_func, batched=True)
        class_counts = train_df['labels'].value_counts(sort=False).tolist()
        
    if not test_df.empty:
        test_dataset = Dataset.from_pandas(test_df[cols])
        if tokenize:
            test_dataset = test_dataset.map(tok_func, batched=True)
    if validation and not val_df.empty:
        val_dataset = Dataset.from_pandas(val_df[cols])
        if tokenize:
            val_dataset = val_dataset.map(tok_func, batched=True)

    # --- Case 1: tokenized mode ---
    if tokenize:    
    # Set output format
        if model_name == "text_only" or "_concat" in model_name or "llama" in model_name:
            columns = ["input_ids", "attention_mask", "labels", "index"]
        elif model_name == "context_all_embed" or model_name == "graph_context_all":
            columns = [
                "text_input_ids", "text_attention_mask",
                "post_input_ids", "post_attention_mask",
                "url_input_ids", "url_attention_mask",
                "parent_input_ids", "parent_attention_mask",
                "labels", "index"
            ]            
        elif "_embed" in model_name :  # context_embed
            columns = [
                "text_input_ids", "text_attention_mask",
                "ctx_input_ids", "ctx_attention_mask",
                "labels", "index"
            ]
        else:
            columns = ["input_ids", "attention_mask", "labels", "index"]

        if train_dataset:
            train_dataset.set_format("torch", columns=columns)
        if test_dataset:
            test_dataset.set_format("torch", columns=columns)
        if val_dataset:
            val_dataset.set_format("torch", columns=columns)

        # Collator logic
        if "_embed" in model_name and model_name != "context_all_embed":
            def collate_dual(batch):
                return {
                    'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                    'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                    'ctx_input_ids': torch.stack([x['ctx_input_ids'] for x in batch]),
                    'ctx_attention_mask': torch.stack([x['ctx_attention_mask'] for x in batch]),
                    'labels': torch.stack([x['labels'] for x in batch]),
                    'index': torch.stack([x['index'] for x in batch]),
                }
            data_collator = collate_dual
        elif model_name == "context_all_embed":
            def collate_four(batch):
                return {
                    'text_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                    'text_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                    'post_input_ids': torch.stack([x['post_input_ids'] for x in batch]),
                    'post_attention_mask': torch.stack([x['post_attention_mask'] for x in batch]),
                    'url_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                    'url_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                    'parent_input_ids': torch.stack([x['text_input_ids'] for x in batch]),
                    'parent_attention_mask': torch.stack([x['text_attention_mask'] for x in batch]),
                    'labels': torch.stack([x['labels'] for x in batch]),
                    'index': torch.stack([x['index'] for x in batch]),
                }
            data_collator = collate_four
        else:
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    else:
        # Just return the dicts with raw text + labels + index
        def collate_raw(batch):
            return {
                'text': [x['text'] for x in batch],
                'labels': torch.tensor([x['labels'] for x in batch]),
                'index': torch.tensor([x['index'] for x in batch])
            }
        data_collator = collate_raw


    train_loader, test_loader, val_loader = None, None, None

    if train_dataset:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler, collate_fn=data_collator)
    
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator) if validation else None

    return train_loader, test_loader, val_loader, class_counts
