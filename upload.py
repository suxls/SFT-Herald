"""Upload the fine-tuned LoRA adapter (or merged model) to Hugging Face Hub.

Usage:
    # Upload LoRA adapter only (small, fast, recommended)
    python upload.py --repo your-username/Qwen3-8B-Herald-SFT

    # Merge LoRA into base model and upload full weights
    python upload.py --repo your-username/Qwen3-8B-Herald-SFT --merge

Prerequisites:
    huggingface-cli login
"""

import argparse

import torch
from peft import AutoPeftModelForCausalLM, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_DIR = "./final_model"


def upload_adapter(repo_id: str, private: bool = False):
    """Push only the LoRA adapter weights (~200 MB)."""
    print(f"Loading adapter from {ADAPTER_DIR} ...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        ADAPTER_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print(f"Pushing adapter to {repo_id} ...")
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)
    print("Done — adapter uploaded.")


def upload_merged(repo_id: str, private: bool = False):
    """Merge LoRA into the base model and push full weights (~16 GB)."""
    print(f"Loading base model {BASE_MODEL} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading adapter from {ADAPTER_DIR} and merging ...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print(f"Pushing merged model to {repo_id} ...")
    model.push_to_hub(repo_id, private=private)
    tokenizer.push_to_hub(repo_id, private=private)
    print("Done — merged model uploaded.")


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id, e.g. your-username/Qwen3-8B-Herald-SFT",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA into base model before uploading (larger, but standalone)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    args = parser.parse_args()

    if args.merge:
        upload_merged(args.repo, private=args.private)
    else:
        upload_adapter(args.repo, private=args.private)


if __name__ == "__main__":
    main()
