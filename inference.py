"""Test the fine-tuned model with the GRPO-compatible format.

Usage:
    python inference.py
    python inference.py --prompt "Prove that for all natural numbers n, n + 0 = n"
    python inference.py --merge   # test with merged model instead of adapter
"""

import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_DIR = "./final_model"

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

Now formalize the following:"""

DEFAULT_PROMPT = (
    "For categories C and D with zero morphisms, let X, Y be objects in C, "
    "f : X → Y a morphism in C, and c a cokernel cofork of f in C. If G is a "
    "functor from C to D that preserves zero morphisms, then the composition of "
    "the map from the parallel pair (G.f, 0) on the left walking parallel pair "
    "homomorphism followed by the homomorphism of the identity isomorphism on the "
    "object corresponding to the first element of the walking parallel pair in D "
    "equals the composition of the homomorphism of the identity isomorphism on the "
    "object corresponding to the zero element of the walking parallel pair in D "
    "followed by the map of the parallel pair (f, 0) ⋄ G on the left walking "
    "parallel pair homomorphism in D."
)


def load_model(merge: bool = False):
    """Load the fine-tuned model (adapter or merged)."""
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    if merge:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model = model.merge_and_unload()
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, nl_statement: str) -> str:
    """Run inference in the GRPO-compatible format."""
    user_content = json.dumps({"nl_statement": nl_statement}, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )

    generated = output_ids[0][inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Test the Herald SFT model")
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Natural language math statement to formalize",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Use merged model instead of base + adapter",
    )
    args = parser.parse_args()

    print("Loading model ...")
    model, tokenizer = load_model(merge=args.merge)

    print(f"\n{'='*60}")
    print(f"INPUT: {args.prompt}")
    print(f"{'='*60}\n")

    response = generate(model, tokenizer, args.prompt)
    print(f"OUTPUT:\n{response}")

    # Try to parse the structured output
    if "####" in response:
        json_str = response.split("####", 1)[1].strip()
        try:
            parsed = json.loads(json_str)
            print(f"\n{'='*60}")
            print("PARSED OUTPUT:")
            print(f"  header:           {parsed.get('header', '')}")
            print(f"  formal_statement: {parsed.get('formal_statement', '')}")
        except json.JSONDecodeError:
            print("\n(Could not parse JSON from output)")


if __name__ == "__main__":
    main()
