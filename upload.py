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
import os

import torch
from huggingface_hub import HfApi
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ADAPTER_DIR = "./final_model"


def upload_adapter(repo_id: str, adapter_dir: str, private: bool = False):
    """Push only the LoRA adapter weights (~200 MB) by uploading the folder directly."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set. Run: export HF_TOKEN=hf_...")
    api = HfApi(token=token)
    print(f"Creating repo {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    print(f"Uploading adapter folder {adapter_dir} → {repo_id} ...")
    api.upload_folder(
        folder_path=adapter_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print("Done — adapter uploaded.")


def upload_merged(repo_id: str, adapter_dir: str, private: bool = False):
    """Merge LoRA into the base model and push full weights (~16 GB)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is not set. Run: export HF_TOKEN=hf_...")
    print(f"Loading base model {BASE_MODEL} ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Loading adapter from {adapter_dir} and merging ...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    print(f"Pushing merged model to {repo_id} ...")
    model.push_to_hub(repo_id, private=private, token=token)
    tokenizer.push_to_hub(repo_id, private=private, token=token)
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
        "--checkpoint",
        type=str,
        default=DEFAULT_ADAPTER_DIR,
        help="Path to the adapter/checkpoint directory (default: ./final_model)",
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
        upload_merged(args.repo, adapter_dir=args.checkpoint, private=args.private)
    else:
        upload_adapter(args.repo, adapter_dir=args.checkpoint, private=args.private)


if __name__ == "__main__":
    main()
