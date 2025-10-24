import torch
import torch_geometric.transforms as T
from torch_geometric.nn import RGCNConv, GraphConv, GATConv, to_hetero
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DistilBertModel,
    LlamaForSequenceClassification,
    LlamaForCausalLM,
    LlamaTokenizer,
    PretrainedConfig,
    RobertaModel
)

#from llama_cpp import Llama

from utils.construct_graph import get_graph, get_hetero_graph


all_model_names = [
    "text_only", # Text classifier over the comment text
    "text_post_concat", # Text classifier over the comment text [SEP] post title
    "com_text_concat", # Text classifier over the account name [SEP] comment text
    "com_text_post_concat", # Text classifier over the account name [SEP] comment text [SEP] post title
    "com_type_text_concat", # Text classifier over the account name page type [SEP] comment text
    "com_type_text_post_concat", # Text classifier over the account name page type [SEP] comment text [SEP] post title
    "domain_text_concat", # Text classifier over the domain name [SEP] comment text
    "source_text_concat", # Text classifier over the source type [SEP] comment text
    "theme_text_concat", # Text classifier over the theme [SEP] comment text
    "domain_source_text_concat", # Text classifier over the domain name source type [SEP] comment text
    "domain_theme_text_concat", # Text classifier over the domain name theme [SEP] comment text
    "parent_text_concat", # text SEP parent_text 
    "article_text_concat", # text SEP post_title SEP post_description
    "message_text_concat", # text SEP post_message
    "context_all_text_concat",

    "post_text_embed", # Generates embeddings for the comment text and post title, combine them with a FC layer, then classify
    "parent_text_embed", 
    "article_text_embed",
    "message_text_embed",
    "com_text_embed",
    "domain_text_embed",
    "context_all_text_embed", 

    "context_all_embed",
    
    "graph_context_all",
    "llama",
    "llama_cpp"
    ]
    
all_base_pretrained_models = [
    "bert-base-uncased",        # BERT (English, uncased)
    "bert-base-cased",          # BERT (English, cased)
    "roberta-base",             # RoBERTa (English)
    "xlm-roberta-base",         # XLM-R (Multilingual)
    "allenai/longformer-base-4096",  # Longformer (English)
    "answerdotai/ModernBERT-base",  # Modern-BERT (English)
    "answerdotai/ModernBERT-large", # Modern-BERT (English)
    "almanach/camembertv2-base", # CamemBERT v2 (French)
    "almanach/camembert-base", # CamemBERT v1 (French)
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct"
]

# DistilBERT Classifier model 
class DistilBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistilBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
class RoBERTaHateClassifier(nn.Module):
    def __init__(self, roberta_model_name="roberta-base", hidden_dim=256, output_dim=1, dropout_prob=0.1):
        super(RoBERTaHateClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        self.fc1 = nn.Linear(self.roberta.config.hidden_size, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token is at index 0
        x = torch.relu(self.fc1(cls_output))
        x = self.dropout(x)
        # Note: we use F.binary_cross_entropy_with_logits as the criterion for classification 
        # hence, we should NOT apply a softmax or sigmoid after the output layer
        return self.fc2(x)

# Simple Text Classifier model 
class SimpleTextClassifier(torch.nn.Module):

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=1, dropout=0.1):
        super(SimpleTextClassifier, self).__init__()
        
        # Linear layer to map input embeddings to hidden dimension
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.1)        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x= self.dropout(x)        
        x = self.fc2(x)        
        x = torch.sigmoid(x)
        return x


class BERTConcat(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, trim="affordance"):
        super(BERTConcat, self).__init__()
        device = get_device()
        self.trim = trim
        
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"
        print(f"initializing Text Concat model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
        
        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        
        # Load pre-trained BERT model for sequence classification with the custom config
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(device)
        
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def forward(self, context_texts, target_text, labels, max_length=4096):
        device = get_device()
        # Combine context texts and target text using the [SEP] token
        input_text = self.prepare_input_2(context_texts, target_text, max_length)
        
        # Tokenize and encode the concatenated input text
        encodings = self.tokenizer(
            input_text,
            padding='max_length',  # Pads up to the max length
            truncation=True,       # Truncate context if needed
            max_length=max_length,        # BERT's max input size
            return_tensors='pt',   # Return PyTorch tensors
        )
        encodings = encodings.to(device)
        shape_encodings = encodings["input_ids"].shape
        print(f"input ids shape for {self.model_name}'s tokenizer: {shape_encodings}")
        # Handle model-specific inputs
        if self.model_name == "longformer":
            # For Longformer, include global attention
            global_attention_mask = torch.zeros_like(encodings["attention_mask"])
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token
        
            model_output = self.text_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                global_attention_mask=global_attention_mask, 
                labels=labels
            )  # Shape: [#comments, 100, hidden_dim]
        else:
            # For BERT, RoBERTa, and XLM-R
            model_output = self.text_model(
                token_type_ids=encodings.get("token_type_ids", None),
                attention_mask=encodings["attention_mask"],
                input_ids=encodings["input_ids"], 
                labels=labels
            )  # Shape: [#comments, 100, hidden_dim]

        return model_output
    
    def prepare_input(self, context_texts, target_text, max_length=512):
        # Prepare the concatenated input text with [SEP] tokens
        # Assuming context_texts is a list of strings
        combined_context = " [SEP] ".join(context_texts)

        # estimate of number of tokens to keep.
        max_ctx_len = max_length - int(len(" [SEP] " + target_text) / 4)
        if max_ctx_len > 0:
            combined_context = combined_context[:max_ctx_len]
        else: 
            combined_context = ""
        
        # Concatenate context and target text
        input_text = combined_context + " [SEP] " + target_text
        
        return input_text

    def prepare_input_2(self, context_texts, target_text, max_length=512):
        # Prepare the concatenated input text with [SEP] tokens
        # Assuming context_texts is a list of strings
        combined_context = " [SEP] ".join(context_texts)
        input_text = target_text + " [SEP] " + combined_context
        return input_text



class BERTReactClass(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.2):
        super(BERTReactClass, self).__init__()
        device = get_device()
        
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"
        print(f"initializing Reac Classifier model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
        
        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        
        # Load pre-trained BERT model for sequence classification with the custom config
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(device)
        
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    def forward(self, reaction_texts, labels, max_length=512):
        device = get_device()
        # Combine reaction texts using the [SEP] tokens
        input_text = self.prepare_input(reaction_texts)
        
        # Tokenize and encode the concatenated input text
        encodings = self.tokenizer(
            input_text,
            padding='max_length',  # Pads up to the max length
            truncation=True,       # Truncate context if needed
            max_length=max_length,        # BERT's max input size
            return_tensors='pt',   # Return PyTorch tensors
        )
        encodings = encodings.to(device)
        # Handle model-specific inputs
        if self.model_name == "longformer":
            # For Longformer, include global attention
            global_attention_mask = torch.zeros_like(encodings["attention_mask"])
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token
        
            model_output = self.text_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                global_attention_mask=global_attention_mask, 
                labels=labels
            )  # Shape: [#comments, 100, hidden_dim]
        else:
            # For BERT, RoBERTa, and XLM-R
            model_output = self.text_model(
                token_type_ids=encodings.get("token_type_ids", None),
                attention_mask=encodings["attention_mask"],
                input_ids=encodings["input_ids"], 
                labels=labels
            )  # Shape: [#comments, 100, hidden_dim]

        return model_output
    

    def prepare_input(self, reactions):
        # Prepare the concatenated input text with [SEP] tokens
        # Assuming reactions is a list of strings
        combined_context = " [SEP] ".join(reactions)
        return combined_context


# Context-Embed model described in https://aclanthology.org/2024.lrec-main.740/
class ContextEmbed(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, trim="affordance"):
        super(ContextEmbed, self).__init__()
        device = get_device()
        self.trim = trim
        
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"
        print(f"initializing ContextEmbed model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
        self.layernorm = nn.LayerNorm(768).to(device)
        
        # Define custom configurations for BERT, and context models
        self.config = AutoConfig.from_pretrained(
            "bert-base-uncased",
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        
        # Load pre-trained BERT model for sequence classification with the custom config
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            config=self.config
        ).to(device)
        
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.fc = torch.nn.Linear(1536, 768).to(device)


    def forward(self, data, target_text, labels, max_length=512):
        device = get_device()
        data = data.to('cpu')
        mask = data["y_mask"]
        _, _, conv_indices_to_keep, my_new_mask_idx = get_graph(data.x_text, mask, with_temporal_edges=False, undirected=False, trim=self.trim)

        context_texts = []
        for i in conv_indices_to_keep:
            if i != my_new_mask_idx:
                context_texts.append(data.x_text[i][0]['body'])
        
        # Combine context texts using the [SEP] token
        combined_context = " [SEP] ".join(context_texts)
        
        # Tokenize target text and combined context
        target_encodings = self.tokenizer(
            target_text,
            padding='max_length',  # Pads up to the max length
            truncation=True,       # Truncate context if needed
            max_length=max_length,        # BERT's max input size
            return_tensors='pt',   # Return PyTorch tensors
        )
        target_encodings = target_encodings.to(device)

        context_encodings = self.tokenizer(
            combined_context,
            padding='max_length',  # Pads up to the max length
            truncation=True,       # Truncate context if needed
            max_length=max_length,        # BERT's max input size
            return_tensors='pt',   # Return PyTorch tensors
        )
        context_encodings = context_encodings.to(device)

        # Encode message (word embeddings from embedding layer)
        target_output = self.text_model.bert(
                token_type_ids=target_encodings.get("token_type_ids", None),
                attention_mask=target_encodings["attention_mask"],
                input_ids=target_encodings["input_ids"]
            ).last_hidden_state[:, 0, :] # Shape: [#comments, 100, hidden_dim]
        context_output = self.text_model.bert(
                token_type_ids=context_encodings.get("token_type_ids", None),
                attention_mask=context_encodings["attention_mask"],
                input_ids=context_encodings["input_ids"]
            ).last_hidden_state[:, 0, :] # Shape: [#comments, 100, hidden_dim]

        # Normalize before concatenation
        context_norm = self.layernorm(context_output)  # Normalize context
        target_norm = self.layernorm(target_output)  # Normalize message embeddings

        # Concatenate context and target embeddings
        combined_embeddings = torch.cat([context_norm, target_norm], dim=-1)
        print(f"combined embedding shape {combined_embeddings.shape}")

        # Project down to hidden_dim
        combined_embeddings = self.fc(combined_embeddings)

        # Apply activation function
        combined_embeddings = F.gelu(combined_embeddings)

        # Residual connection
        #combined_embeddings = combined_embeddings + target_norm  
        print(f"combined embedding shape after residual {combined_embeddings.shape}")
        

        # Classification head
        #output_representation = encoder_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        out = self.text_model.classifier(combined_embeddings)
        return out


class PostTextEmb(nn.Module):
    def __init__(self, pretrained_model_name='camembert-base', num_classes=2,
                 hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3):
        super().__init__()
        self.device = get_device()
        self.model_name = pretrained_model_name

        # Load config with specified dropout values
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Load full model to get both encoder and classifier
        full_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(self.device)

        # Extract encoder and classifier head
        self.encoder = full_model.base_model  # works for BERT, RoBERTa, CamemBERT, etc.

        self.reduce_fc = nn.Linear(1536, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_text; CLS_title] → 768
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, self.config.num_labels)
        ).to(self.device)
        
    def forward(self, batch):
        """
        batch must contain:
            - text_input_ids
            - text_attention_mask
            - ctx_input_ids
            - ctx_attention_mask
        """
        print(f'inside forward, batch keys; {batch.keys()}')
        text_input_ids = batch["text_input_ids"].to(self.device)
        text_attention_mask = batch["text_attention_mask"].to(self.device)
        ctx_input_ids = batch["ctx_input_ids"].to(self.device)
        ctx_attention_mask = batch["ctx_attention_mask"].to(self.device)
        
        # Encode both text and title independently
        text_outputs = self.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        ctx_outputs = self.encoder(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask
        )

        # Extract [CLS] token embeddings
        text_cls = text_outputs.last_hidden_state[:, 0, :]   # (B, 768)
        ctx_cls = ctx_outputs.last_hidden_state[:, 0, :] # (B, 768)

        # Concatenate and reduce
        combined = torch.cat([text_cls, ctx_cls], dim=1)   # (B, 1536)
        reduced = self.reduce_fc(combined)                   # (B, 768)

        # Final classification (reusing pretrained classifier head)
        logits = self.classifier(reduced)                    # (B, num_classes)

        return logits
    

class AllContextEmb(nn.Module):
    def __init__(self, pretrained_model_name='camembert-base', num_classes=2,
                 hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3, add_parent=True):
        super().__init__()
        self.device = get_device()
        self.model_name = pretrained_model_name
        self.add_parent = add_parent

        # Load config with specified dropout values
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Load full model to get both encoder and classifier
        full_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(self.device)

        # Extract encoder and classifier head
        self.encoder = full_model.base_model  # works for BERT, RoBERTa, CamemBERT, etc.
        if add_parent:
            self.reduce_ctx = nn.Linear(2304, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_post; CLS_url; CLS_parent] → 768
        else:
            self.reduce_ctx = nn.Linear(1536, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_post; CLS_url; CLS_parent] → 768

        self.reduce_fc = nn.Linear(1536, 768).to(self.device) # Intermediate reduction layer: concatenated [CLS_text; CLS_ctx] → 768
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(768, self.config.num_labels)
        ).to(self.device)
        
    def forward(self, batch):
        """
        batch must contain:
            - text_input_ids
            - text_attention_mask
            - post_input_ids
            - post_attention_mask
            - url_input_ids
            - url_attention_mask
        """
        add_parent = self.add_parent
        text_input_ids = batch["text_input_ids"].to(self.device)
        text_attention_mask = batch["text_attention_mask"].to(self.device)
        post_input_ids = batch["post_input_ids"].to(self.device)
        post_attention_mask = batch["post_attention_mask"].to(self.device)
        url_input_ids = batch["url_input_ids"].to(self.device)
        url_attention_mask = batch["url_attention_mask"].to(self.device)

        # Encode both text and title independently
        text_outputs = self.encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        post_outputs = self.encoder(
            input_ids=post_input_ids,
            attention_mask=post_attention_mask
        )
        url_outputs = self.encoder(
            input_ids=url_input_ids,
            attention_mask=url_attention_mask
        )

        # Extract [CLS] token embeddings
        text_cls = text_outputs.last_hidden_state[:, 0, :]   # (B, 768)
        post_cls = post_outputs.last_hidden_state[:, 0, :] # (B, 768)
        url_cls = url_outputs.last_hidden_state[:, 0, :] # (B, 768)

        combined_ctx = torch.cat([post_cls, url_cls], dim=1)   # (B, 1536)

        # If add_parent, do the same for the parent_comment
        if add_parent:
            parent_input_ids = batch["parent_input_ids"].to(self.device)
            parent_attention_mask = batch["parent_attention_mask"].to(self.device)
            parent_outputs = self.encoder(
                input_ids=parent_input_ids,
                attention_mask=parent_attention_mask
            )
            parent_cls = parent_outputs.last_hidden_state[:, 0, :] # (B, 768)
            combined_ctx = torch.cat([post_cls, url_cls, parent_cls], dim=1)   # (B, 2304)
        
        # Concatenate and reduce the context
        reduced_ctx = self.reduce_ctx(combined_ctx)         # (B, 768)

        # Concatenate and reduce text + context
        combined = torch.cat([text_cls, reduced_ctx], dim=1)   # (B, 1536)
        reduced = self.reduce_fc(combined)                  # (B, 768)

        # Final classification (reusing pretrained classifier head)
        logits = self.classifier(reduced)                    # (B, num_classes)

        return logits

class GraphContextModel(nn.Module):
    def __init__(self, pretrained_model_name="camembert-base", num_classes=2,
                 hidden_dropout_prob=0.3, attention_probs_dropout_prob=0.3,
                 hidden_size=768, num_heads=4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob
        )

        # Load encoder backbone
        full_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(self.device)

        self.encoder = full_model.base_model  # CamemBERT, BERT, etc.

        # Graph Attention Layer (1 layer + ELU)
        self.gat = GATConv(hidden_size, hidden_size, heads=num_heads, dropout=0.3)
        self.act = nn.ELU()

        # Reduction and classifier
        self.reduce_fc = nn.Linear(hidden_size + hidden_size * num_heads, hidden_size).to(self.device)  # concat(text_cls, graph_text_emb)
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.hidden_dropout_prob),
            nn.Linear(hidden_size, num_classes)
        ).to(self.device)

    def forward(self, batch):
        """
        batch must contain:
            - text_input_ids, text_attention_mask
            - post_input_ids, post_attention_mask
            - url_input_ids, url_attention_mask
            - parent_input_ids, parent_attention_mask
        """
        # Encode each input separately
        text_outputs = self.encoder(input_ids=batch["text_input_ids"], attention_mask=batch["text_attention_mask"])
        post_outputs = self.encoder(input_ids=batch["post_input_ids"], attention_mask=batch["post_attention_mask"])
        url_outputs = self.encoder(input_ids=batch["url_input_ids"], attention_mask=batch["url_attention_mask"])
        parent_outputs = self.encoder(input_ids=batch["parent_input_ids"], attention_mask=batch["parent_attention_mask"])

        # CLS embeddings
        text_cls = text_outputs.last_hidden_state[:, 0, :]    # (B, 768)
        post_cls = post_outputs.last_hidden_state[:, 0, :]    # (B, 768)
        url_cls = url_outputs.last_hidden_state[:, 0, :]      # (B, 768)
        parent_cls = parent_outputs.last_hidden_state[:, 0, :]# (B, 768)

        # Build graph per batch element
        batch_size = text_cls.size(0)
        logits_list = []

        for i in range(batch_size):
            # 4 nodes: [text, post, url, parent]
            node_feats = torch.stack([text_cls[i], post_cls[i], url_cls[i], parent_cls[i]], dim=0)  # (4, 768)

            # Edges: connect text (0) with others (1,2,3), bidirectional
            edge_index = torch.tensor([
                [0, 0, 0, 1, 2, 3],
                [1, 2, 3, 0, 0, 0]
            ], dtype=torch.long, device=self.device)  # (2, E)

            # Apply GAT
            g_emb = self.gat(node_feats, edge_index)  # (4, hidden_size * heads)
            g_emb = self.act(g_emb)

            # Take contextualized embedding of main text node (index 0)
            g_text_emb = g_emb[0]

            # Concatenate with original text_cls
            combined = torch.cat([text_cls[i], g_text_emb], dim=-1)  # (1536,)
            reduced = self.reduce_fc(combined)  # (768,)
            logit = self.classifier(reduced)   # (num_classes,)
            logits_list.append(logit.unsqueeze(0))

        logits = torch.cat(logits_list, dim=0)  # (B, num_classes)
        return logits

class SimpleGraphModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads=1):
        super(SimpleGraphModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads)
        self.fc = torch.nn.Linear(hidden_channels, 1)  # Output one value for binary classification
    
    
    def forward(self, x, edge_indices):   
        # Save the original batch size
        #batch_size = x.size(0)
        device = x.device
        edge_indices = edge_indices.to(device)
    
        x = x.squeeze(0)
        edge_indices = edge_indices.squeeze(0)
        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        
        edge_indices = edge_indices.t().contiguous()
  
        x = self.gat1(x, edge_indices)
        x = F.relu(x)
 
        # GAT layer 2
        x = self.gat2(x, edge_indices)
        x = F.relu(x)

        # Classification layer (binary classification)
        out = self.fc(x) 
        out = out.squeeze(-1)        
        return out

# Credits, code from DialogueGCN https://github.com/declare-lab/conv-emotion/blob/master/DialogueGCN/model.py
class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_relations, max_seq_len, hidden_size=64, dropout=0.5, no_cuda=False):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNetwork, self).__init__()
        
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations, num_bases=30)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        #self.matchatt = MatchingAttention(num_features+hidden_size, num_features+hidden_size, att_type='general2')
        self.linear   = nn.Linear(num_features+hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.smax_fc  = nn.Linear(hidden_size, num_classes)
        self.no_cuda = no_cuda 

    def forward(self, x, edge_index, edge_norm, edge_type, seq_lengths, umask, nodal_attn, avec):
        
        out = self.conv1(x, edge_index, edge_type, edge_norm)
        out = self.conv2(out, edge_index)
        emotions = torch.cat([x, out], dim=-1)
        log_prob = classify_node_features(emotions, seq_lengths, umask, self.matchatt, self.linear, self.dropout, self.smax_fc, nodal_attn, avec, self.no_cuda)
        return log_prob

def classify_node_features(emotions, seq_lengths, umask, matchatt_layer, linear_layer, dropout_layer, smax_fc_layer, nodal_attn, avec, no_cuda):
    """
    Function for the final classification, as in Equation 7, 8, 9. in the paper.
    """

    if nodal_attn:

        #emotions = attentive_node_features(emotions, seq_lengths, umask, matchatt_layer, no_cuda)
        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return torch.cat([hidden[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])

        log_prob = F.log_softmax(hidden, 2)
        log_prob = torch.cat([log_prob[:, j, :][:seq_lengths[j]] for j in range(len(seq_lengths))])
        return log_prob

    else:

        hidden = F.relu(linear_layer(emotions))
        hidden = dropout_layer(hidden)
        hidden = smax_fc_layer(hidden)

        if avec:
            return hidden

        log_prob = F.log_softmax(hidden, 1)
        return log_prob

# New graph model
class GATModel(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2):
        super(GATModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_1, dropout=dropout_1)
        self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification

        # Determine model type based on the pretrained model name
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"

        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        # Load pre-trained text model for sequence classification with the custom config
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        ).to(device)

        self.model = model

        # Extract model-specific components
        if self.model_name == "longformer":
            self.text_model = model.longformer
        elif self.model_name == "bert":
            self.text_model = model.bert
        elif self.model_name == "roberta":
            self.text_model = model.roberta
        elif self.model_name == "xlm-roberta":
            self.text_model = model.roberta  # XLM-Roberta shares architecture with Roberta

        # Other shared components
        self.text_pooler = getattr(self.text_model, "pooler", None)  # Pooler (if available)
        self.node_classifier = model.classifier
        self.dropout = getattr(model, "dropout", None)

                
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    
    

    def forward(self, data):   
        # Save the original batch size
        #batch_size = x.size(0)
        x = data.x
        device = get_device()
        edge_indices = data.edge_index.permute(1, 0)
        mask = data["y_mask"]

        # Tokenized input data
        x_token_type_ids = x.get("token_type_ids", None)  # Optional for models without token types
        x_attention_mask = x["attention_mask"]  # Required for all models
        x_input_ids = x["input_ids"]  # Input IDs

        # Handle model-specific inputs
        if self.model_name == "longformer":
            # For Longformer, include global attention
            global_attention_mask = torch.zeros_like(x_attention_mask)
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token

            model_output = self.text_model(
                input_ids=x_input_ids,
                attention_mask=x_attention_mask,
                global_attention_mask=global_attention_mask
            ).last_hidden_state  # Shape: [#comments, 100, hidden_dim]
        else:
            # For BERT, RoBERTa, and XLM-R
            model_output = self.text_model(
                token_type_ids=x_token_type_ids,
                attention_mask=x_attention_mask,
                input_ids=x_input_ids
            ).last_hidden_state  # Shape: [#comments, 100, hidden_dim]

        
        g_data = Data(x=model_output, edge_index=edge_indices.t().contiguous())
        x, edge_indices = g_data.x, g_data.edge_index
        x = x.to(device)
        edge_indices = edge_indices.to(device)
        # Extract CLS token embeddings for node classification
        cls_embeddings = model_output[:, 0, :]  # Shape: [#comments, hidden_dim]

         # GAT layer 1
        x = self.gat1(cls_embeddings, edge_indices)
        x = F.elu(x)
         # GAT layer 2
        x = self.gat2(x, edge_indices)
        x_gemb = x[mask, :]
        x_emb = cls_embeddings[mask, :]

        concat_out = torch.cat([x_gemb, x_emb], dim=1)
        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        

        # Classification layer (binary classification)
        out = self.fc(concat_out) 
        #out = self.text_pooler(out)
        #out = self.bert_dropout(out)
        out = self.node_classifier(out)
        return out

# New graph model
class GATTest(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.4, dropout_2=0.4, undirected=True, temp_edges=False, num_layers=2, pretrained_model_name='bert-base-uncased', num_classes=2, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2, trim="affordance", new_trim=False):
        super(GATTest, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.trim = trim
        self.new_trim = new_trim
        if num_layers == 1:
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_2, dropout=dropout_1)
        else:
            self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_1, dropout=dropout_1)
            if num_layers == 2:
                self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
            else:
                self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
            if num_layers == 3:
                self.gat3 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
            if num_layers == 4:
                self.gat3 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
                self.gat4 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
            if num_layers == 5:
                self.gat3 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
                self.gat4 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_1)
                self.gat5 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2)
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification
        # Determine model type based on the pretrained model name
        self.model_name = "bert"
        if "longformer" in pretrained_model_name:
            self.model_name = "longformer"
        elif "xlm-roberta" in pretrained_model_name:
            self.model_name = "xlm-roberta"
        elif "roberta" in pretrained_model_name:
            self.model_name = "roberta"
        elif "answerdotai" in pretrained_model_name:
            self.model_name = "modernbert"
        print(f"initializing gat-test model with {self.model_name} text model, pretrained model name is {pretrained_model_name}")
       

        # Define a custom configuration for BERT with dropout
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,  # Number of classes for classification
            hidden_dropout_prob=hidden_dropout_prob,  # Dropout for hidden layers
            attention_probs_dropout_prob=attention_probs_dropout_prob  # Dropout for attention layers
        )
        # Load pre-trained text model for sequence classification with the custom config
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=self.config
        )

        # Extract model-specific components
        if self.model_name == "longformer":
            self.text_model = model.longformer
        elif self.model_name == "bert":
            self.text_model = model.bert
        elif self.model_name == "roberta":
            self.text_model = model.roberta
        elif self.model_name == "xlm-roberta":
            self.text_model = model.roberta  # XLM-Roberta shares architecture with Roberta

        # Other shared components
        #self.text_pooler = getattr(self.text_model, "pooler", None)  # Pooler (if available)
        self.node_classifier = model.classifier
        #self.dropout = getattr(model, "dropout", None)
                
        # Tokenizer for encoding text inputs
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        self.undirected = undirected
        self.temp_edges = temp_edges
        self.num_layers = num_layers
        print('Inside GATTEST init num layers is ', self.num_layers)
    

    def forward(self, data, max_length=512, gpu=True):   
        # Save the original batch size
        #batch_size = x.size(0)
        data = data.to('cpu')
        mask = data["y_mask"]
        _, edges_dic_num, conv_indices_to_keep, my_new_mask_idx = get_graph(data.x_text, mask, with_temporal_edges=self.temp_edges, undirected=self.undirected, trim=self.trim, new_trim=self.new_trim)
        assert len(edges_dic_num.keys()) <= 1, "length of edges dic num is greater than 1"
        edge_list = []
        for k in edges_dic_num.keys():
            edge_list = edges_dic_num[k]
            edge_list = sorted(edge_list, key=lambda x: (x[0], x[1]))
            edge_list = torch.tensor(edge_list)
        x = data.x
        y = data.y
        #edge_indices = data.edge_index.permute(1, 0)
        #are_equal = torch.equal(edge_list, edge_indices)
        #print(f"Are the tensors equal? {are_equal}")
        # YES. The undirected option without temporal edges is equal to edge_indices (the edge list returned by the mDT codebase.)

        texts = []
        for i in conv_indices_to_keep:
            texts.append(data.x_text[i][0]['body'])
        print(f"Number of comments kept in conversation: {len(conv_indices_to_keep)}")

        labels = y.long()
        device_bert = get_device()
        if gpu: 
            device_bert = torch.device('cuda:2')
        print(f"Bert device is {device_bert}")
        self.text_model = self.text_model.to(device_bert)

        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt').to(device_bert)

        # Handle model-specific inputs
        if self.model_name == "longformer":
            # For Longformer, include global attention
            global_attention_mask = torch.zeros_like(encodings["attention_mask"])
            global_attention_mask[:, 0] = 1  # Apply global attention to the first token

            model_output = self.text_model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                global_attention_mask=global_attention_mask
            ).last_hidden_state  # Shape: [#comments, 100, hidden_dim]
        else:
            # For BERT, RoBERTa, and XLM-R
            model_output = self.text_model(
                token_type_ids=encodings.get("token_type_ids", None),
                attention_mask=encodings["attention_mask"],
                input_ids=encodings["input_ids"]
            ).last_hidden_state  # Shape: [#comments, 100, hidden_dim]

        model_output = model_output.to('cpu')
        g_data = Data(x=model_output, edge_index=edge_list.t().contiguous())        
        x, edge_list = g_data.x, g_data.edge_index
        cls_embeddings = model_output[:, 0, :] 
        edge_indices = edge_list
        
        device_graph = get_device() 
        if gpu:
            device_graph = torch.device('cuda:1')
        print(f"Bert graph is {device_graph}")

        # if there are no edges in the graph, ignore the GAT layers
        if edge_indices.numel() == 0:
            x = cls_embeddings
        else:
            # GAT layer 1
            self.gat1 = self.gat1.to(device_graph)
            cls_embeddings = cls_embeddings.to(device_graph)
            edge_indices = edge_indices.to(device_graph)
            x = self.gat1(cls_embeddings, edge_indices)
            x = F.elu(x)
            if self.num_layers > 1:
                # GAT layer 2
                self.gat2 = self.gat2.to(device_graph)
                x = self.gat2(x, edge_indices)
                if self.num_layers == 3:
                    self.gat3 = self.gat3.to(device_graph)
                    # GAT layer 3
                    x = F.elu(x)
                    x = self.gat3(x, edge_indices)
                elif self.num_layers == 4:
                    self.gat3 = self.gat3.to(device_graph)
                    self.gat4 = self.gat4.to(device_graph)
                    # GAT layer 3
                    x = F.elu(x)
                    x = self.gat3(x, edge_indices)
                    # GAT layer 4
                    x = F.elu(x) 
                    x = self.gat4(x, edge_indices)
                elif self.num_layers == 5:
                    self.gat3 = self.gat3.to(device_graph)
                    self.gat4 = self.gat4.to(device_graph)
                    self.gat5 = self.gat5.to(device_graph)
                    # GAT layer 3
                    x = F.elu(x)
                    x = self.gat3(x, edge_indices)
                    # GAT layer 4
                    x = F.elu(x) 
                    x = self.gat4(x, edge_indices)
                    # GAT layer 5
                    x = F.elu(x)
                    x = self.gat5(x, edge_indices)
        device_classification = get_device()
        if gpu:
            device_classification = torch.device('cuda:0')   
        print(f"Bert class is {device_classification}")            
        x_gemb = x[my_new_mask_idx].unsqueeze(0).to(device_classification)
        x_emb = cls_embeddings[my_new_mask_idx].unsqueeze(0).to(device_classification)

        concat_out = torch.cat([x_gemb, x_emb], dim=1)

        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]

        # Classification layer (binary classification)
        self.fc = self.fc.to(device_classification)
        self.node_classifier = self.node_classifier.to(device_classification)
        out = self.fc(concat_out) 
        #out = self.text_pooler(out)
        #out = self.bert_dropout(out)
        if "roberta" in self.model_name or "longformer" == self.model_name:
            out = out.unsqueeze(1)  # Add sequence dimension: [batch_size, seq_length=1, hidden_dim]
            out = self.node_classifier(out).squeeze(1)  # Remove sequence dimension after classification
        else:
            out = self.node_classifier(out)  # Directly use for BERT
        return out



# graphModel is the internal graph network used by both heterogenous and homogenous graph models.
class graphModel(torch.nn.Module):
    def __init__(self, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6):
        super(graphModel, self).__init__()
        #self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=num_heads_1, dropout=dropout_1, add_self_loops=False)
        self.gat2 = GATConv(hidden_channels * num_heads_1, hidden_channels, heads=num_heads_2, add_self_loops=False)

    def forward(self, x, edge_list):   
         # GAT layer 1
        x = self.gat1(x, edge_list)
        x = F.elu(x)
         # GAT layer 2
        x = self.gat2(x, edge_list)
        return x


class HeteroGAT(torch.nn.Module):
    def __init__(self, model=GATTest, in_channels=768, hidden_channels=768, num_heads_1=8, num_heads_2=1, dropout_1=0.6, dropout_2=0.6, undirected=True, temp_edges=False):
        super(HeteroGAT, self).__init__()
        self.fc = torch.nn.Linear(1536, 768)  # Output one value for binary classification
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            hidden_dropout_prob=0.3,
            attention_probs_dropout_prob=0.3,
        )
        # enable gradient checkpointing to reduce CUDA memory and send more work to compute 
        # this was added after I encountered CUDA OOM errors when training HeteroGAT on CAD data. 
        bert.gradient_checkpointing_enable()
        self.text_model = bert.bert
        self.text_pooler = self.text_model.pooler
        self.node_classifier = bert.classifier
        self.bert_dropout = bert.dropout
        self.undirected = undirected
        self.temp_edges = temp_edges

        self.graph_model = graphModel()

    def forward(self, data):   
        device = get_device()

        # use model parallelism to split the data and steps on different devices 
        # Move the data object to the CPU at the beginning
        data = data.to('cpu')
        mask = data["y_mask"]

        num_comment_nodes, comments_edges_dic_num, num_users, user_to_comments_edges, conv_indices_to_keep, my_new_mask_idx = get_hetero_graph(data.x_text, mask, with_temporal_edges=self.temp_edges)
        assert conv_indices_to_keep[my_new_mask_idx] == mask.nonzero(as_tuple=True)[0], "error: the new index should be equivalent to the old one."
        user_to_comments_edges = torch.tensor(user_to_comments_edges, dtype=torch.long).permute(1,0).to('cpu')
        comments_to_user_edges = user_to_comments_edges.flip(0).to('cpu')

        hetero_graph = HeteroData()

        # Add node indices:
        hetero_graph["comment"].node_id = torch.arange(num_comment_nodes)
        hetero_graph["user"].node_id = torch.arange(num_users)

        # Add edge indices:
        if len(user_to_comments_edges) > 0:
            hetero_graph["user", "posts", "comment"].edge_index = user_to_comments_edges #.t().contiguous()
        if len(comments_to_user_edges) > 0:    
            hetero_graph["comment", "posted_by", "user"].edge_index = comments_to_user_edges
            
        assert len(comments_edges_dic_num.keys()) == 1, "There should be only one dictionary key"
        comments_edge_list = []
        for k in comments_edges_dic_num.keys():
            comments_edge_list = comments_edges_dic_num[k]
            comments_edge_list = sorted(comments_edge_list, key=lambda x: (x[0], x[1]))
            if len(comments_edge_list) > 0:
                comments_edge_list = torch.tensor(comments_edge_list, dtype=torch.long).permute(1,0).to('cpu')
        if len(comments_edge_list) > 0:
            hetero_graph["comment", "replies", "comment"].edge_index = comments_edge_list #.t().contiguous()
  
        # We also need to make sure to add the reverse edges from movies to users
        # in order to let a GNN be able to pass messages in both directions.
        # We can leverage the `T.ToUndirected()` transform for this from PyG:
        if self.undirected:
            hetero_graph = T.ToUndirected()(hetero_graph)
        x = data.x

        # Model //sm: BERT Model runs on cuda:2
        device_bert = torch.device('cuda:2')
        texts = []
        for i in conv_indices_to_keep:
            texts.append(data.x_text[i][0]['body'])
        y = data.y

        labels = y.long().to(device)
        encodings = self.tokenizer(texts, truncation=True, padding='max_length', max_length=300, return_tensors='pt').to(device_bert)

        self.text_model = self.text_model.to(device_bert)

        bert_output = self.text_model(
            token_type_ids=encodings["token_type_ids"],
            attention_mask=encodings["attention_mask"],
            input_ids=encodings["input_ids"],
        ).last_hidden_state # [#comments, 100, 768]

        #x_token_type_ids = x["token_type_ids"][conv_indices_to_keep].to(device_bert) # [#comments, 100]
        #x_attention_mask = x["attention_mask"][conv_indices_to_keep].to(device_bert) # [#comments, 100]
        #x_input_ids = x["input_ids"][conv_indices_to_keep].to(device_bert) # [#comments, 100]
        
        #bert_output = self.text_model(
        #    token_type_ids=x_token_type_ids,
        #    attention_mask=x_attention_mask,
        #    input_ids=x_input_ids,
        #).last_hidden_state # [#comments, 100, 768]
        cls_embeddings = bert_output[:, 0, :] 

        # Add node features: in CPU
        hetero_graph["comment"].x = cls_embeddings.to('cpu')
        # must be the same feature size for each node type
        hetero_graph["user"].x = torch.ones((num_users, 768)).to('cpu') 
        #TODO(celia): add scores as edge features
    
        # Model //sm: Graph Model on cuda:1
        device_graph = torch.device('cuda:1')
        hetero_graph = hetero_graph.to(device_graph, non_blocking=True)
        self.graph_model = self.graph_model.to(device_graph)
        model = to_hetero(self.graph_model, hetero_graph.metadata(), aggr='sum')
        model = model.to(device_graph)
        x = model(hetero_graph.x_dict, hetero_graph.edge_index_dict)

        comments = x['comment']
        x_gemb = comments[my_new_mask_idx].unsqueeze(0).to(device_graph) 
        x_emb = cls_embeddings[my_new_mask_idx].unsqueeze(0).to(device_graph)
        concat_out = torch.cat([x_gemb, x_emb], dim=1)

        # SHAPE OF X: [#vertices, 768=input_dim] --> [#vertices, 64=hidden_dim] --> [#vertices, 1] --> [#vertices]
        
        # Model //sm: Classification layer (binary classification) on cuda:0
        device_classification = torch.device('cuda:0')
        concat_out = concat_out.to(device_classification)
        self.fc = self.fc.to(device_classification)
        self.node_classifier = self.node_classifier.to(device_classification)
        out = self.fc(concat_out) 
        out = self.node_classifier(out)    
        return out
        
def get_model(args):
    model_name = args.model_name
    assert model_name in all_model_names, "Invalid model name: {}".format(model_name)
    pretrained_model_name = args.pretrained_model_name
    assert pretrained_model_name in all_base_pretrained_models, "Invalid model name: {}".format(pretrained_model_name)
    device = get_device()
    # Instantiate your model
    model = ""

    if model_name == "text_only" or "_concat" in model_name:
        custom_config = AutoConfig.from_pretrained(
            pretrained_model_name,
            num_labels=2,                    
            hidden_dropout_prob=args.hidden_dropout_prob,         
            attention_probs_dropout_prob=args.attention_probs_dropout_prob 
        )     
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            config=custom_config
        )
    elif "text_embed" in model_name:
        print('text_embed in model name ==> PostTextEmb model')
        model = PostTextEmb(
            pretrained_model_name=pretrained_model_name,  
            num_classes=2,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_probs_dropout_prob
        )
    elif model_name == "context_all_embed":
        model = AllContextEmb(
            pretrained_model_name=pretrained_model_name,  
            num_classes=2,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_probs_dropout_prob,
            add_parent=False
        )
    elif model_name == "graph_context_all":
        model = GraphContextModel(
            pretrained_model_name=pretrained_model_name,  
            num_classes=2,
            hidden_dropout_prob=args.hidden_dropout_prob, 
            attention_probs_dropout_prob=args.attention_probs_dropout_prob
        )

    elif "llama" in model_name:
        # 2️⃣ 4-bit quantization config (saves memory & disk)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = LlamaForCausalLM.from_pretrained(
            pretrained_model_name,
            device_map="auto",       # automatically shards across all available GPUs
            quantization_config=bnb_config,
            offload_folder="/tmp/llama_offload",
            torch_dtype=torch.bfloat16,  # low memory dtype
        )

    if "llama" not in model_name:
        model = model.to(device)
    return model


def get_device(rank=None):
    if rank is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        print(f"Using CUDA device {rank} for distributed training")
        return device

    if torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device('mps')

    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device('cuda')

    print("Using CPU")
    return torch.device('cpu')

def log_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"Allocated Memory: {allocated_memory:.2f} GB")
    print(f"Reserved Memory: {reserved_memory:.2f} GB")
