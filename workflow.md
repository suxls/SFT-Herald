# Herald SFT → GRPO Training Workflow

End-to-end guide: SFT cold-start on Qwen3-8B with Herald_statements, then GRPO
reinforcement learning with a Lean 4 compiler reward.

---

## 0. Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Login to HuggingFace (for dataset download and model upload)
huggingface-cli login

# (Optional) Login to Weights & Biases for logging
wandb login
```

**Hardware requirements:**

| Setup | VRAM | Estimated SFT time (2 epochs, 580k) |
|---|---|---|
| 1x RTX 4090 / A5000 (24 GB) — QLoRA | 24 GB | ~16-24 hours |
| 1x A100-40GB — QLoRA | 40 GB | ~10-16 hours |
| 1x A100-80GB — QLoRA | 80 GB | ~8-12 hours |
| 1x H100-80GB — QLoRA | 80 GB | ~5-8 hours |
| 2x A100-80GB — QLoRA | 160 GB | ~4-6 hours |

> **How the estimate works:** The 580k dataset with packing (avg ~4 examples per
> 2048-token window) yields ~145k packed sequences per epoch. With effective
> batch size 16 (bs=2 × grad_accum=8), that's ~9k steps/epoch × 2 epochs =
> ~18k steps. Each step takes ~1.5-3s depending on GPU → 8-15 hours typical.

---

## 1. Dataset Split Strategy

The 580k Herald_statements dataset should be split for the two-phase pipeline:

```
Herald_statements (580k)
├── SFT training set   (90%)  →  ~522k examples
├── SFT validation set (1%)   →  ~5.8k examples
└── GRPO held-out set  (9%)   →  ~52k examples   ← reserved for RL
```

**Why this split:**

- **SFT (90%)** — The model needs to learn the structured output format
  (`#### {"header": ..., "formal_statement": ...}`) and the NL→Lean translation
  mapping. More data = more robust format adherence, which is critical for
  downstream GRPO parsing.

- **GRPO (9%)** — Held-out examples the model hasn't memorized. During GRPO, the
  reward comes from the Lean 4 compiler (does the generated code type-check?),
  so the model learns to produce *correct* formalizations, not just
  syntactically plausible ones. Using unseen examples prevents reward hacking
  on memorized outputs.

- **Validation (1%)** — Monitor SFT loss convergence and catch overfitting early.

To apply this split, edit `train.py` line 119:

```python
# Replace the simple train_test_split with a three-way split:
dataset = dataset.train_test_split(test_size=0.10, seed=42)
grpo_dataset = dataset["test"]                        # 10% held out
sft_dataset = dataset["train"].train_test_split(       # 90% → 89% train + 1% val
    test_size=0.011, seed=42
)
# Save the GRPO set for later
grpo_dataset.save_to_disk("./data/grpo_held_out")

# Use sft_dataset["train"] and sft_dataset["test"] for SFT training
```

Then in the trainer section, use `sft_dataset["train"]` and `sft_dataset["test"]`.

> **Alternative:** If you want maximum SFT quality and plan to reuse the same
> data for GRPO (common in practice), keep the current 99/1 split for SFT and
> use the full 580k again for GRPO. The RL signal (compiler check) is
> orthogonal to the SFT memorization risk, so overlap is acceptable.

---

## 2. Phase 1 — SFT (Cold Start)

**Goal:** Teach the base model the structured input/output format and basic
NL→Lean translation ability.

### Quick start

```bash
# Single GPU
python train.py

# Multi-GPU (recommended if available)
accelerate launch --multi_gpu --num_processes=NUM_GPUS train.py
```

### What to monitor

Watch these in wandb (or stdout logs):

| Metric | Healthy range | Red flag |
|---|---|---|
| `train/loss` | Drops from ~3.0 → 0.3-0.8 | Plateaus above 1.5 |
| `eval/loss` | Tracks train loss closely | Diverges from train (overfitting) |
| `train/learning_rate` | Cosine curve, peaks at 2e-4 | — |

### Smoke test

After training completes (~8-24h depending on hardware):

```bash
python inference.py --prompt "Prove that for all natural numbers n, n + 0 = n"
```

Expected output format:
```
#### {"header": "import Mathlib", "formal_statement": "theorem nat_add_zero (n : ℕ) : n + 0 = n := by sorry"}
```

If the model reliably produces valid `####`-prefixed JSON, SFT is working.

---

## 3. Phase 2 — GRPO with vLLM (Reinforcement Learning)

**Goal:** Optimize the SFT model to produce Lean 4 code that actually
type-checks, using compiler feedback as reward.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                    │
│                                                         │
│  ┌──────────┐    generate    ┌──────────────┐           │
│  │  vLLM    │ ─────────────→ │  Parse ####  │           │
│  │  (SFT    │    N samples   │  JSON output │           │
│  │  model)  │    per prompt  │              │           │
│  └──────────┘                └──────┬───────┘           │
│                                     │                   │
│                                     ▼                   │
│                              ┌──────────────┐           │
│                              │  Lean 4      │           │
│                              │  Compiler    │           │
│                              │  (reward)    │           │
│                              └──────┬───────┘           │
│                                     │                   │
│                                     ▼                   │
│                              ┌──────────────┐           │
│                              │  GRPO Loss   │           │
│                              │  (group      │           │
│                              │  relative    │           │
│                              │  policy opt) │           │
│                              └──────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Reward function design

```python
def reward_fn(response: str, ground_truth: str) -> float:
    """
    Reward for GRPO.  Parse the #### JSON, compile with Lean 4.

    Returns:
        1.0  — type-checks successfully
        0.5  — valid Lean 4 syntax but doesn't type-check (partial)
        0.1  — valid JSON format but invalid Lean 4
        0.0  — malformed output (no #### or invalid JSON)
    """
    if "####" not in response:
        return 0.0

    json_str = response.split("####", 1)[1].strip()
    try:
        parsed = json.loads(json_str)
        header = parsed["header"]
        formal = parsed["formal_statement"]
    except (json.JSONDecodeError, KeyError):
        return 0.0

    lean_code = header + "\n" + formal
    result = lean4_typecheck(lean_code)  # your Lean 4 REPL / lake check

    if result.success:
        return 1.0
    elif result.is_syntax_valid:
        return 0.5
    else:
        return 0.1
```

### GRPO training sketch (with TRL)

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir="./results_grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,            # much lower than SFT
    num_generations=8,             # G=8 samples per prompt for group scoring
    max_completion_length=1024,
    bf16=True,
    use_vllm=True,                 # serve with vLLM for fast generation
    vllm_gpu_utilization=0.7,
)

grpo_trainer = GRPOTrainer(
    model="./final_model",         # start from SFT checkpoint
    args=config,
    train_dataset=grpo_dataset,    # the 52k held-out set
    reward_funcs=[reward_fn],
    processing_class=tokenizer,
)

grpo_trainer.train()
```

### Lean 4 environment setup

GRPO needs a Lean 4 compiler accessible from Python. Options:

1. **LeanDojo** — Python bindings for Lean 4, easiest to integrate
2. **lean4-repl** — Lightweight REPL, good for batch checking
3. **Docker container** — Run `lake build` in an isolated container per sample
4. **Lean LSP** — Use the language server protocol for incremental checking

---

## 4. Upload to HuggingFace

```bash
# Upload LoRA adapter only (~200 MB, users load base + adapter)
python upload.py --repo your-username/Qwen3-8B-Herald-SFT

# Or merge into base model and upload standalone (~16 GB)
python upload.py --repo your-username/Qwen3-8B-Herald-SFT --merge

# For GRPO model later:
python upload.py --repo your-username/Qwen3-8B-Herald-GRPO --merge
```

---

## 5. Full Timeline Estimate

| Phase | Duration | Hardware | Output |
|---|---|---|---|
| Setup & deps | 30 min | Any | Environment ready |
| SFT training | 8-24h | 1x A100 80GB | `./final_model/` |
| Smoke test | 15 min | Same GPU | Verify format output |
| Upload SFT | 10-30 min | Internet | HF repo with adapter |
| GRPO setup | 2-4h | — | Lean 4 env + reward fn |
| GRPO training | 24-72h | 2x A100 80GB | `./results_grpo/` |
| Upload GRPO | 10-30 min | Internet | Final HF repo |
| **Total** | **~2-5 days** | | |

---

## 6. Tips & Troubleshooting

**SFT phase:**
- If loss doesn't drop below 1.0 after 2k steps, check that `format_chat()`
  is producing valid messages (print a few samples).
- If you get OOM, reduce `per_device_train_batch_size` to 1 and increase
  `gradient_accumulation_steps` to 16.
- To disable wandb: set `report_to="none"` in `SFTConfig` or
  `WANDB_DISABLED=true python train.py`.
- Remove `attn_implementation="flash_attention_2"` from `train.py` if your
  GPU doesn't support FlashAttention 2 (needs Ampere+).

**GRPO phase:**
- Start with `num_generations=4` if memory is tight, increase to 8-16 for
  better gradient estimates.
- The Lean 4 compiler is the bottleneck — batch compile calls and use
  timeouts (10-30s per check).
- If reward is near-zero everywhere, the SFT model isn't good enough yet.
  Train SFT for more epochs or on more data before attempting GRPO.
- Use `learning_rate=5e-6` or lower — GRPO should refine, not destabilize.

**General:**
- Always save checkpoints. SFT crashes at step 8000 of 18000 means you
  resume from step 8000, not restart.
- The `packing=True` flag significantly speeds up SFT. Disable it only if
  you see training instability.
