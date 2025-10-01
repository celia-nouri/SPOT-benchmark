import os
os.environ["VLLM_USE_FLEX_ATTENTION"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoConfig
import argparse

# ---------------------------
# Load Prompt
# ---------------------------
def load_prompt(prompt_file):
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------
# Inference Function
# ---------------------------
def classify_batch(prompts, llm, sampling_params):
    max_len = llm.llm_engine.model_config.max_model_len

    # Truncate each prompt if it's too long
    truncated_prompts = []
    for p in prompts:
        if len(p) > max_len:
            truncated_prompts.append(p[:max_len])
        else:
            truncated_prompts.append(p)

    outputs = llm.generate(truncated_prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


# ---------------------------
# Main Eval Loop
# ---------------------------
def run_inference(df, model_name, llm, sampling_params, prompt_file, batch_size=8):
    results = []
    PROMPT_TEMPLATE = load_prompt(prompt_file)
    print(f"Using PROMPT_TEMPLATE: {PROMPT_TEMPLATE}\n")
    if "mistral" in model_name.lower():
        PROMPT_TEMPLATE =  "<s>[INST] " + PROMPT_TEMPLATE + " [/INST]</s>"
    if "qwen" in model_name.lower():
         PROMPT_TEMPLATE =  "[INST] " + PROMPT_TEMPLATE + " [/INST]"
    for i in tqdm(range(0, len(df), batch_size), desc="Running inference"):
        batch_texts = df["text"].iloc[i:i+batch_size].tolist()
        if "text_context" in prompt_file:
            batch_posts = df["post_message"].iloc[i:i+batch_size].tolist()
            batch_descriptions = df["post_description"].iloc[i:i+batch_size].tolist()
            batch_parents = df["parent_text"].iloc[i:i+batch_size].tolist()
            batch_url_titles = df["post_title"].iloc[i:i+batch_size].tolist()
            batch_prompts = [
                PROMPT_TEMPLATE.format(text=t, title=p, url_title=u, description=d,parent_comment=c)
                for t, p, u, d, c in zip(batch_texts, batch_posts, batch_url_titles, batch_descriptions, batch_parents)
            ]
        elif "context" in prompt_file:
            batch_urls = df["post_url"].iloc[i:i+batch_size].tolist()
            batch_prompts = [
                PROMPT_TEMPLATE.format(text=t, url=u)
                for t, u in zip(batch_texts, batch_urls)
            ]
        else:
            batch_prompts = [PROMPT_TEMPLATE.format(text=t) for t in batch_texts]
        print(f"BATCH TEXTS: {batch_texts}")
        preds = classify_batch(batch_prompts, llm, sampling_params)
        print(f"PREDS: {preds}")
        results.extend(preds)
    return results

def normalize_label(pred: str) -> int:
    """Convert raw model output to {0,1,-1}."""
    if pred.startswith("1"):
        return 1
    elif pred.startswith("0"):
        return 0
    else:
        return -1

# ---------------------------
# Call function
# ---------------------------
def run_llm(args):
    print(f"Opening data path: {args.data_path}")
   
    #df = pd.read_csv(args.data_path, low_memory=False)

    df = pd.read_csv(args.data_path, sep=';', low_memory=False, quotechar='"')

    
    if "text" not in df.columns:
        raise ValueError("CSV must contain 'text' columns.")
    df["text"] = df["text"].fillna("").astype(str)

    # vLLM sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,  # deterministic outputs
        max_tokens=5,
    )

    # ---------------------------
    # Load Model
    # ---------------------------
    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name)

    # Determine max context length from config (fallback = 8192)
    #max_len = getattr(config, "max_position_embeddings", 8192)

    tokenizer_mode = "auto"
    if "mistral" in model_name.lower():
        tokenizer_mode = "mistral"

    # Precision → "auto" lets vLLM decide based on GPU (half precision if supported)
    dtype = "auto"

    print(f"Loading {model_name} with max_len=2048, dtype={dtype}, tokenizer_mode={tokenizer_mode}")

    llm = LLM(
        model=model_name,
        tokenizer_mode='auto',
        dtype='float16',
        max_model_len=2048,
        tensor_parallel_size=2, 
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # (optional) squeeze more memory
    )

    # Example eval dataframe
#    eval_df = pd.DataFrame({
#        "text": [
#            "C’est complètement faux, fake news !",
#            "Merci pour l’info 👍",
#            "Photoshop raté, montage mal fait."
#        ]
#    })

    predictions = run_inference(df, args.model_name, llm, sampling_params, args.prompt_file, args.batch_size)

    df["predicted_label"] = [normalize_label(p) for p in predictions]

    print(df)
    # Save if needed
    print(f"Writing to output path: {args.output_path}")
    df.to_csv(args.output_path, index=False)




def parse_args(parser):
    # Data args
    parser.add_argument("--data_path", type=str, default="data/error_analysis_testset_fake.csv") #"data/interro1_Manon_tous.csv")
    parser.add_argument('--size', type=str, default='large', help='the size of the dataset, can take one of the following values: ["small", "medium", "large", "small-1000", "cad"]')
    parser.add_argument('--validation', type=bool, default=True, help='rather or not to use a validation set for model tuning')

    # Model args
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help='the model to use') 
    # mistralai/Mistral-7B-Instruct-v0.2
    # meta-llama/Llama-3.2-3B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
  
    # Hyper params
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    # Inference argument
    parser.add_argument('--inference-only', action='store_true', help='Run inference only and write predictions to CSV')
    parser.add_argument('--output_path', type=str, default="error_analysis_testset_qwen.csv", help='Path to save inference results (CSV with predicted scores and labels)')
    parser.add_argument("--prompt_file", type=str, default="models/prompts/prompt_llama.txt", help="Filename of the prompt template in prompts/")
    return parser.parse_args()


# ---------------------------
# Main call
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier for point d'arrêt detection")
    args = parse_args(parser)
    run_llm(args)
