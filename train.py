import json
import re

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ──────────────────────────────────────────────
# 1. Load dataset
# ──────────────────────────────────────────────
dataset = load_dataset("FrenzyMath/Herald_statements", split="train")

# Optional: use a subset for faster experimentation
# dataset = dataset.shuffle(seed=42).select(range(50_000))

# ──────────────────────────────────────────────
# 2. System prompt & formatting
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a Lean4 auto-formalization engine.

## Task
Given a statement of a math problem, translate it into formal Lean 4 (v4.11.0) code with Mathlib.
Do not write any proof steps. Use "sorry" as a placeholder for proofs.
All statements must end with ":= by sorry" or ":= sorry".

## Input Format
{"nl_statement": "<natural language math problem>"}

## Output Format
Output must start with #### followed by a JSON object:
#### {"header": "<import statements>", "formal_statement": "<lean code> := by sorry"}

## Important Guidelines
1. Use Lean 4 syntax (not Lean 3): "by" instead of "begin...end"
2. Import Mathlib appropriately (e.g., "import Mathlib" or specific modules)
3. For inequalities, use proper comparison operators: <, ≤, >, ≥
4. For set membership, use: x ∈ Set.Ioo a b (open interval), Set.Icc a b (closed)
5. Common Lean 4 notations: ℝ (reals), ℕ (naturals), ∀, ∃, →, ↔
6. Name theorems descriptively (e.g., theorem inequality_problem_1)
7. If the problem involves "prove" or "show", formalize the goal after the colon
8. If variables aren't specified, infer reasonable types (ℝ for algebra, ℕ for number theory)

## Examples

Example 1:
User: {"nl_statement": "Solve for $x$ in the given inequality: $x^2-2x-24<0$"}
Assistant: #### {"header": "import Mathlib", "formal_statement": "theorem example_1 (x : ℝ) : x^2 - 2*x - 24 < 0 ↔ x ∈ Set.Ioo (-4) 6 := by sorry"}

Example 2:
User: {"nl_statement": "Prove that for all natural numbers n, n + 0 = n"}
Assistant: #### {"header": "import Mathlib", "formal_statement": "theorem nat_add_zero (n : ℕ) : n + 0 = n := by sorry"}

Example 3:
User: {"nl_statement": "Show that if a divides b and b divides c, then a divides c for positive integers"}
Assistant: #### {"header": "import Mathlib", "formal_statement": "theorem dvd_trans (a b c : ℕ) (hab : a ∣ b) (hbc : b ∣ c) : a ∣ c := by sorry"}

Example 4:
User: {"nl_statement": "For positive reals a, b, c with abc=1, prove $(a+b)(b+c)(c+a) \\geq 8$"}
Assistant: #### {"header": "import Mathlib", "formal_statement": "theorem am_gm_extension (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : (a + b) * (b + c) * (c + a) ≥ 8 := by sorry"}

Now formalize the following:"""

# Regex to find the first Lean 4 declaration keyword, including optional modifiers
DECL_RE = re.compile(
    r"(?:(?:noncomputable|protected|private|unsafe|partial)\s+)*"
    r"(?:theorem|lemma|def|instance|example|class|structure|inductive|abbrev)\b"
)


def parse_formal_statement(formal_stmt: str) -> tuple[str, str]:
    """Split a full formal statement into (header, declaration).

    The header contains import/open lines; the declaration starts at
    the first Lean 4 declaration keyword (theorem, lemma, def, etc.).
    """
    m = DECL_RE.search(formal_stmt)
    if m:
        header = formal_stmt[: m.start()].strip()
        statement = formal_stmt[m.start() :].strip()
    else:
        header = ""
        statement = formal_stmt.strip()
    return header, statement


def format_chat(example):
    """Format each example into the GRPO-compatible chat template.

    User message:  {"nl_statement": "<informal statement>"}
    Assistant msg:  #### {"header": "<imports>", "formal_statement": "<lean code>"}
    """
    header, stmt = parse_formal_statement(example["formal_statement"])

    user_content = json.dumps(
        {"nl_statement": example["informal_statement"]}, ensure_ascii=False
    )
    assistant_content = "#### " + json.dumps(
        {"header": header, "formal_statement": stmt}, ensure_ascii=False
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    return {"messages": messages}


dataset = dataset.map(format_chat, remove_columns=dataset.column_names, num_proc=8)

# ──────────────────────────────────────────────
# Three-way split: 90% SFT train / 1% SFT val / 9% GRPO held-out
# ──────────────────────────────────────────────
split = dataset.train_test_split(test_size=0.10, seed=42)
grpo_dataset = split["test"]
sft_split = split["train"].train_test_split(test_size=0.011, seed=42)

grpo_dataset.save_to_disk("./data/grpo_held_out")
print(f"SFT train: {len(sft_split['train']):,}  |  "
      f"SFT val: {len(sft_split['test']):,}  |  "
      f"GRPO held-out: {len(grpo_dataset):,} (saved to ./data/grpo_held_out)")

# ──────────────────────────────────────────────
# 3. Load tokenizer
# ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen3-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ──────────────────────────────────────────────
# 4. Quantization config (QLoRA — 4-bit)
# ──────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ──────────────────────────────────────────────
# 5. Load model
# ──────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model = prepare_model_for_kbit_training(model)

# ──────────────────────────────────────────────
# 6. LoRA config
# ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# ──────────────────────────────────────────────
# 7. Training arguments
# ──────────────────────────────────────────────
training_args = SFTConfig(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    max_seq_length=2048,
    packing=True,
    dataset_kwargs={"add_special_tokens": False},
    report_to="wandb",
    seed=42,
)

# ──────────────────────────────────────────────
# 8. Create trainer & train
# ──────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_split["train"],
    eval_dataset=sft_split["test"],
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.train()

# ──────────────────────────────────────────────
# 9. Save LoRA adapter locally
# ──────────────────────────────────────────────
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
