"""
Week 1, Lab 2 (v3): AutoTokenizer Internals, Qwen3 Thinking/Tool Templates & Left-Padding Masks
Model: Qwen/Qwen3-4B (Fast CPU/Tokenizer-only lab — downloads ~5MB tokenizer config, runs in <2s).

Key MLOps & FDE Concepts Covered:
1. Subword Tokenization (Byte-Level BPE) & token-to-ID mapping
2. Qwen3 Chat Templates (`apply_chat_template`):
   - Reasoning mode toggle (`enable_thinking=True` vs `False` -> `<think>...</think>`)
   - Native Agentic Tool Calling (`tools=` schema injection -> `<tools>...</tools>`)
3. Left-Padding vs. Right-Padding for Decoder-Only (Causal LM) batched inference
4. Attention Masks (`1` = attend, `0` = ignore pad token)
"""

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-4B"


def inspect_subword_tokenization(tokenizer: AutoTokenizer) -> None:
    """Demonstrate how raw text, code, and numbers split into subwords."""
    print("\n" + "=" * 78)
    print("1. SUBWORD TOKENIZATION & VOCABULARY INTERNALS (Qwen3 Byte-Level BPE)")
    print("=" * 78)
    print(f"Tokenizer Class : {tokenizer.__class__.__name__}")
    print(
        f"Vocab Size      : {tokenizer.vocab_size:,} base tokens "
        f"(total with added special tokens = {len(tokenizer):,})"
    )
    print(
        f"Special Tokens  : bos={tokenizer.bos_token!r} ({tokenizer.bos_token_id}), "
        f"eos={tokenizer.eos_token!r} ({tokenizer.eos_token_id}), "
        f"pad={tokenizer.pad_token!r} ({tokenizer.pad_token_id})"
    )

    samples = [
        "Kubernetes PodCrashLoopBackOff",
        "def calculate_vram(params_b: float) -> float:\n    return params_b * 2.0",
        "Transaction ID: 984710293847",
    ]

    for sample in samples:
        token_ids = tokenizer.encode(sample, add_special_tokens=False)
        subwords = [tokenizer.decode([tid]) for tid in token_ids]
        print(f"\nRaw String : {sample!r}")
        print(f"Token Count: {len(token_ids)}")
        print(f"Token IDs  : {token_ids}")
        print(f"Subwords   : {subwords}")


def inspect_chat_and_tool_templates(tokenizer: AutoTokenizer) -> None:
    """Show exact control tokens injected by `apply_chat_template` for thinking & tool calling."""
    print("\n" + "=" * 78)
    print("2. QWEN3 CHAT TEMPLATES: THINKING MODE & NATIVE TOOL CALLING")
    print("=" * 78)

    messages = [
        {"role": "system", "content": "You are a strict enterprise FinOps assistant."},
        {"role": "user", "content": "Check current GPU utilization in us-central1."},
    ]

    # A. Compare enable_thinking=False vs enable_thinking=True
    for thinking_flag in (False, True):
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_flag,
        )
        print(f"\n[A] Rendered Prompt (enable_thinking={thinking_flag}):")
        print("-" * 50)
        print(repr(rendered))

    # B. Inspect Native Function/Tool Calling Template (`tools=` parameter)
    def get_gpu_utilization(region: str) -> dict:
        """Query live GPU cluster utilization for a given cloud region.

        Args:
            region: Cloud region identifier (e.g., 'us-central1', 'asia-south1').
        """
        return {"region": region, "utilization_pct": 84.5}

    rendered_with_tools = tokenizer.apply_chat_template(
        messages,
        tools=[get_gpu_utilization],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("\n[B] Rendered Prompt with Native `tools=[get_gpu_utilization]`:")
    print("-" * 50)
    print(rendered_with_tools)
    print("-" * 50)


def inspect_batched_padding_and_masks(tokenizer: AutoTokenizer) -> None:
    """
    Demonstrate why Decoder-only LLMs require LEFT padding during batched generation.
    In autoregressive generation, the next token is predicted from the LAST position
    in the sequence. If right-padded, the last position is a <pad> token!
    """
    print("\n" + "=" * 78)
    print("3. BATCHED INFERENCE: LEFT-PADDING & ATTENTION MASKS")
    print("=" * 78)

    # Configure for decoder-only batched generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_prompts = [
        "Ping.",
        "Explain PagedAttention in vLLM in one short sentence.",
    ]

    batch = tokenizer(
        batch_prompts,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    print(f"Padding Side   : {tokenizer.padding_side!r}")
    print(
        f"input_ids shape: {tuple(batch['input_ids'].shape)} (Batch Size x Sequence Length)"
    )
    print(f"mask shape     : {tuple(batch['attention_mask'].shape)}")

    for i, prompt in enumerate(batch_prompts):
        ids = batch["input_ids"][i].tolist()
        mask = batch["attention_mask"][i].tolist()
        tokens = [tokenizer.decode([tid]) for tid in ids]
        print(f"\nBatch Item {i} ({prompt!r}):")
        print(f"  Tokens        : {tokens}")
        print(f"  Input IDs     : {ids}")
        print(
            f"  Attention Mask: {mask}  <-- 0 masks out left-pad tokens in Self-Attention!"
        )


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inspect_subword_tokenization(tokenizer)
    inspect_chat_and_tool_templates(tokenizer)
    inspect_batched_padding_and_masks(tokenizer)


if __name__ == "__main__":
    main()
