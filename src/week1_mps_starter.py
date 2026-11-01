"""
Week 1, Lab 1 (v3): Production-Grade Local Inference on Apple Silicon (MPS)
Model: Qwen/Qwen3-4B (Apache-2.0, Modern 2026 Dense 4B Model)
Standards: Hugging Face `transformers` v5 (`dtype=`, `DynamicCache`, `sdpa`)

IMPORTANT M4 24GB MEMORY NOTE:
If `Qwen3-Coder-30B-A3B` is currently loaded in LM Studio (~14.5 GB), unload it first
(`lms unload --all` or click Eject in LM Studio) before running this script so PyTorch
MPS has ~8 GB of free Unified Memory for Qwen3-4B in bfloat16.
"""

from dataclasses import dataclass
import logging
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferenceConfig:
    model_id: str = "Qwen/Qwen3-4B"
    device: str = "mps"
    # M4 natively accelerates bfloat16 on MPS (~8.0 GB VRAM for 4B params)
    dtype: torch.dtype = torch.bfloat16
    max_new_tokens: int = 256
    temperature: float = 0.2
    enable_thinking: bool = False  # Toggle True to inspect Qwen3 <think> reasoning tokens


def verify_mps_environment() -> torch.device:
    """Validate that PyTorch is compiled with Metal support and MPS is accessible."""
    if not torch.backends.mps.is_built():
        logger.error("PyTorch was not built with MPS support.")
        sys.exit(1)

    if not torch.backends.mps.is_available():
        logger.error("MPS device not found. Check macOS version and Xcode CLI tools.")
        sys.exit(1)

    logger.info("MPS backend verified (PyTorch %s).", torch.__version__)
    return torch.device("mps")


def log_mps_telemetry(model: torch.nn.Module, stage: str) -> None:
    """Verify parameter residency on the Apple GPU and report Metal memory allocation."""
    param_device = next(model.parameters()).device
    param_dtype = next(model.parameters()).dtype

    # Allocated by PyTorch tensors vs. total reserved by the Metal driver
    current_mem_gb = torch.mps.current_allocated_memory() / (1024**3)
    driver_mem_gb = torch.mps.driver_allocated_memory() / (1024**3)

    logger.info(
        "[%s] Device: %s | Dtype: %s | MPS Tensor Mem: %.2f GB | Metal Driver Mem: %.2f GB",
        stage,
        param_device,
        param_dtype,
        current_mem_gb,
        driver_mem_gb,
    )

    if param_device.type != "mps":
        raise RuntimeError(f"Expected model on 'mps', but found '{param_device}'.")


def load_pipeline_components(
    config: InferenceConfig,
) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and model explicitly onto the MPS device using transformers v5 API."""
    logger.info("Loading tokenizer & model: %s", config.model_id)
    start_time = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    # transformers v5 uses `dtype=` (replaces legacy `torch_dtype=`).
    # attn_implementation="sdpa" routes attention through PyTorch's Metal SDPA kernels.
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=config.dtype,
        device_map=config.device,
        attn_implementation="sdpa",
    )
    model.eval()

    elapsed = time.perf_counter() - start_time
    logger.info("Model loaded in %.2f seconds.", elapsed)
    log_mps_telemetry(model, stage="Post-Load")

    return tokenizer, model


def generate_response(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    config: InferenceConfig,
) -> str:
    """Tokenize chat input, execute MPS-accelerated generation with DynamicCache, and measure throughput."""
    messages = [
        {
            "role": "system",
            "content": "You are a concise Senior AI Systems & Forward Deployed Engineer.",
        },
        {"role": "user", "content": prompt},
    ]

    # 1. Apply Qwen3 chat template (supports `enable_thinking` budget control)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
        return_tensors="pt",
        return_dict=True,
    ).to(config.device)

    input_token_count = inputs["input_ids"].shape[-1]
    logger.info(
        "Input tokenized: %d tokens on device '%s' (enable_thinking=%s).",
        input_token_count,
        inputs["input_ids"].device,
        config.enable_thinking,
    )

    # 2. Initialize explicit transformers v5 DynamicCache to inspect KV-cache state post-generation
    kv_cache = DynamicCache()

    # 3. Synchronize Metal command queue before starting the timer
    torch.mps.synchronize()
    t0 = time.perf_counter()

    # 4. Run forward pass with autograd disabled
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            past_key_values=kv_cache,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.temperature > 0.0,
            temperature=config.temperature if config.temperature > 0.0 else None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
             use_cache=True,
        )

    # Synchronize again so async GPU kernels finish before stopping the clock
    torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    # 5. Slice off the prompt tokens and decode only the newly generated tokens
    generated_ids = outputs[0, input_token_count:]
    num_generated = generated_ids.shape[0]
    tokens_per_sec = num_generated / elapsed if elapsed > 0 else 0.0

    logger.info(
        "Generated %d tokens in %.2fs (%.1f tokens/sec) | KV-Cache Seq Len: %d",
        num_generated,
        elapsed,
        tokens_per_sec,
        kv_cache.get_seq_length(),
    )
    log_mps_telemetry(model, stage="Post-Inference")

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main() -> None:
    config = InferenceConfig()
    verify_mps_environment()

    tokenizer, model = load_pipeline_components(config)

    prompt = (
        "A client CISO asks why we recommend self-hosting a 4B parameter model for internal "
        "ticket triage instead of sending all tickets to an external frontier API. "
        "Give 3 concrete architectural and TCO reasons."
    )
    response = generate_response(prompt, tokenizer, model, config)

    print("\n" + "=" * 70)
    print(f"PROMPT: {prompt}\n")
    print(f"RESPONSE:\n{response.strip()}")
    print("=" * 70 + "\n")

    # Release cached blocks back to the Metal driver
    torch.mps.empty_cache()


if __name__ == "__main__":
    main()
