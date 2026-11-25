"""
Week 1, Lab 3 (v4.0): Transformer Weight Dissection (SVD), Local TRL LoRA Fine-Tuning
& Adapter Trajectory Analysis (Delta_W = (alpha / r) * B @ A) + Hugging Face Hub Publisher

Target Model : Qwen/Qwen3-4B (Apache-2.0) on Apple M4 Metal (`mps`, bfloat16)
Target HF Hub: https://huggingface.co/garg-achal/qwen3-4b-finops-lora-week1

IMPORTANT M4 24GB UNIFIED MEMORY GUARDRAIL:
Before running this script, ensure LM Studio's 30B model is unloaded:
    lms unload --all

Usage:
    # 1. Run full local SVD dissection + LoRA fine-tuning + Delta_W trajectory analysis:
    .venv/bin/python src/week1_svd_lora_lab.py

    # 2. Run AND publish your trained adapter + SVD Model Card to https://huggingface.co/garg-achal:
    .venv/bin/python src/week1_svd_lora_lab.py --push-to-hub
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import math
import time

from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabConfig:
    model_id: str = "Qwen/Qwen3-4B"
    hub_repo_id: str = "garg-achal/qwen3-4b-finops-lora-week1"
    output_dir: str = "outputs/qwen3-4b-finops-lora-week1"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16
    inspect_layer_idx: int = 18  # Middle transformer block in Qwen3-4B (36 layers total)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_steps: int = 15
    learning_rate: float = 2e-4


# ==============================================================================
# PART 1: TRANSFORMER WEIGHT DISSECTION & SVD SPECTRUM ANALYSIS
# ==============================================================================


def effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    """Compute Shannon-entropy Effective Rank (Roy & Vetterli, 2007) of a weight matrix.

    Given singular values sigma_1 >= sigma_2 >= ... >= sigma_min > 0:
        p_i = sigma_i / sum(sigma_j)
        H(p) = -sum(p_i * log(p_i))
        erank(W) = exp(H(p))
    """
    s = singular_values.clamp(min=1e-12)
    probs = s / s.sum()
    entropy = -torch.sum(probs * torch.log(probs)).item()
    return math.exp(entropy)


def dissect_transformer_weights_and_svd(
    model: AutoModelForCausalLM, layer_idx: int
) -> list[dict[str, float | str | tuple[int, int]]]:
    """Inspect Attention (W_q, W_k, W_v, W_o) & SwiGLU MLP projections + SVD spectrum."""
    print("\n" + "=" * 86)
    print(
        f"1. TRANSFORMER WEIGHT DISSECTION & SVD SPECTRUM (Layer {layer_idx} of {model.config.num_hidden_layers})"
    )
    print("=" * 86)

    layer = model.model.layers[layer_idx]
    projections = {
        "self_attn.q_proj": layer.self_attn.q_proj.weight,
        "self_attn.k_proj": layer.self_attn.k_proj.weight,
        "self_attn.v_proj": layer.self_attn.v_proj.weight,
        "self_attn.o_proj": layer.self_attn.o_proj.weight,
        "mlp.gate_proj": layer.mlp.gate_proj.weight,
        "mlp.up_proj": layer.mlp.up_proj.weight,
        "mlp.down_proj": layer.mlp.down_proj.weight,
    }

    q_out, q_in = projections["self_attn.q_proj"].shape
    k_out, _ = projections["self_attn.k_proj"].shape
    gqa_ratio = q_out // k_out
    print(
        f"Grouped-Query Attention (GQA) Geometry: Q_dim={q_out}, KV_dim={k_out} "
        f"--> {gqa_ratio}:1 Q-to-KV Head Ratio ({gqa_ratio}x KV-Cache VRAM reduction!)\n"
    )
    print(
        f"{'Module Name':<20} | {'Shape (d_out, d_in)':<20} | {'sigma_1':>8} | "
        f"{'Top-8 Energy':>12} | {'Eff. Rank / Full':>18}"
    )
    print("-" * 86)

    svd_telemetry: list[dict[str, float | str | tuple[int, int]]] = []
    with torch.no_grad():
        for name, weight in projections.items():
            shape = tuple(weight.shape)
            full_rank = min(shape)
            # Move weight to CPU float32 for exact LAPACK SVD decomposition
            w_fp32 = weight.detach().to(device="cpu", dtype=torch.float32)
            s = torch.linalg.svdvals(w_fp32)

            energy_total = torch.sum(s**2).item()
            top8_energy_pct = (torch.sum(s[:8] ** 2).item() / energy_total) * 100.0
            erank = effective_rank_from_singular_values(s)

            print(
                f"{name:<20} | {str(shape):<20} | {s[0].item():>8.2f} | "
                f"{top8_energy_pct:>11.2f}% | {erank:>8.1f} / {full_rank:<6d}"
            )
            svd_telemetry.append(
                {
                    "module": name,
                    "shape": shape,
                    "sigma_1": round(s[0].item(), 3),
                    "top8_energy_pct": round(top8_energy_pct, 2),
                    "effective_rank": round(erank, 1),
                    "full_rank": full_rank,
                }
            )

    return svd_telemetry


# ==============================================================================
# PART 2: CURATE ENTERPRISE FINOPS DATASET & RUN LOCAL TRL SFT + LORA
# ==============================================================================


def build_finops_training_dataset(tokenizer: AutoTokenizer) -> Dataset:
    """Create a domain-specific B2B LLM FinOps & GPU Incident Triage dataset."""
    raw_examples = [
        {
            "user": "Our AWS g5.2xlarge (24GB A10G) running a 7B FP16 model hits CUDA OOM at 16 concurrent users with 4,096 context length. Diagnose the bottleneck and prescribe the fix.",
            "assistant": (
                "[FINOPS_AUDIT_REPORT]\n"
                "1. Root Cause: 7B FP16 weights consume 14.0 GB VRAM, leaving only ~8.5 GB for KV-Cache and activations. "
                "At 16 concurrent sequences x 4,096 tokens, contiguous FP16 KV-Cache allocation exceeds available VRAM.\n"
                "2. Remediation: Quantize weights to W4A16/AWQ (~4.2 GB VRAM) and serve on vLLM with PagedAttention and FP8 KV-Cache.\n"
                "3. Projected ROI: Reduces base weight VRAM by 70%, increases max concurrency from 16 to 96+ users on the same A10G GPU, cutting unit token cost by 78%."
            ),
        },
        {
            "user": "Why is our batch inference pipeline emitting repetitive garbage tokens when batch_size > 1 on a decoder-only Llama/Qwen model?",
            "assistant": (
                "[FINOPS_AUDIT_REPORT]\n"
                "1. Root Cause: Tokenizer defaults to right-padding (`padding_side='right'`). Because causal decoder-only models sample the next token from the final sequence index (-1), shorter prompts attend to trailing `<pad>` tokens.\n"
                "2. Remediation: Enforce `tokenizer.padding_side = 'left'` and pass `attention_mask` so left-padded positions receive -inf before the attention softmax.\n"
                "3. Projected ROI: Eliminates 100% of batch corruption retries and enables high-throughput continuous batching."
            ),
        },
        {
            "user": "We have 12 enterprise product teams each wanting a custom fine-tuned 4B model. Do we need 12 separate GPU instances?",
            "assistant": (
                "[FINOPS_AUDIT_REPORT]\n"
                "1. Root Cause: Deploying 12 full-weight checkpoints wastes 12x static GPU memory on identical base weights.\n"
                "2. Remediation: Train rank-8 PEFT LoRA adapters (~35 MB each) and deploy a single base model in vLLM with `--enable-lora` dynamic multi-adapter routing.\n"
                "3. Projected ROI: Consolidates 12 GPU nodes into 1 shared HA pair, slashing monthly GPU infrastructure spend by 83%."
            ),
        },
        {
            "user": "Our customer-facing JSON extraction endpoint has a 1,800ms Time-To-First-Token (TTFT) on Qwen3-4B. How do we bring TTFT under 250ms?",
            "assistant": (
                "[FINOPS_AUDIT_REPORT]\n"
                "1. Root Cause: Hybrid reasoning (`enable_thinking=True`) generates hundreds of hidden `<think>...</think>` tokens before emitting the first JSON brace.\n"
                "2. Remediation: Set `enable_thinking=False` in `apply_chat_template` (pre-filling `<think>\\n\\n</think>`) and enforce schema grammar via Outlines.\n"
                "3. Projected ROI: Cuts TTFT from 1,800ms to <190ms and reduces output token billing by 65%."
            ),
        },
    ] * 4  # Repeat 4x (16 samples) for a clean 15-step local convergence demonstration

    formatted_texts = []
    for item in raw_examples:
        messages = [
            {
                "role": "system",
                "content": "You are the Lead AI Systems & FinOps Architect at curons.org. Always respond using the structured [FINOPS_AUDIT_REPORT] format with Root Cause, Remediation, and Projected ROI.",
            },
            {"role": "user", "content": item["user"]},
            {"role": "assistant", "content": item["assistant"]},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
        )
        formatted_texts.append({"text": rendered})

    return Dataset.from_list(formatted_texts)


def run_lora_sft_and_inspect_delta_w(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: LabConfig,
) -> tuple[AutoModelForCausalLM, list[dict[str, float | str | int]], float, float]:
    """Attach LoRA adapters, train with TRL SFTTrainer, and analyze Delta_W = (alpha/r) * B @ A."""
    print("\n" + "=" * 86)
    print(
        f"2. LOCAL TRL SFT + PEFT LORA TRAINING (r={config.lora_r}, alpha={config.lora_alpha}, device={config.device})"
    )
    print("=" * 86)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    peft_model = get_peft_model(model, lora_config)
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    trainable_pct = (trainable_params / all_params) * 100.0
    print(
        f"LoRA Parameters : {trainable_params:,} trainable / {all_params:,} total "
        f"({trainable_pct:.4f}% parameter footprint)"
    )

    dataset = build_finops_training_dataset(tokenizer)

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        max_steps=config.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=config.learning_rate,
        logging_steps=5,
        save_strategy="no",
        dataset_text_field="text",
        max_length=384,
        bf16=(config.dtype == torch.bfloat16 and config.device != "cpu"),
        report_to="none",
        seed=42,
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=tokenizer,
    )

    if config.device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    train_result = trainer.train()
    if config.device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    log_history = [entry for entry in trainer.state.log_history if "loss" in entry]
    initial_loss = log_history[0]["loss"] if log_history else train_result.training_loss
    final_loss = log_history[-1]["loss"] if log_history else train_result.training_loss
    print(
        f"\nTraining Complete in {elapsed:.1f}s | Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}"
    )

    # Save adapter weights locally for inspection and optional HF Hub upload
    peft_model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # ==========================================================================
    # PART 3: INSPECT ADAPTER TENSOR TRAJECTORIES (Delta_W = (alpha / r) * B @ A)
    # ==========================================================================
    print("\n" + "=" * 86)
    print(
        f"3. ADAPTER TENSOR TRAJECTORY & SVD VERIFICATION (Layer {config.inspect_layer_idx}: Delta_W = (alpha/r) * B @ A)"
    )
    print("=" * 86)
    print(
        f"{'Module':<20} | {'||Delta_W||_F':>14} | {'||Delta_W|| / ||W||':>20} | "
        f"{'sigma_1(Delta_W)':>16} | {'Non-Zero Rank':>14}"
    )
    print("-" * 86)

    layer = peft_model.base_model.model.model.layers[config.inspect_layer_idx]
    target_dict = {
        "self_attn.q_proj": layer.self_attn.q_proj,
        "self_attn.k_proj": layer.self_attn.k_proj,
        "self_attn.v_proj": layer.self_attn.v_proj,
        "self_attn.o_proj": layer.self_attn.o_proj,
        "mlp.gate_proj": layer.mlp.gate_proj,
        "mlp.up_proj": layer.mlp.up_proj,
        "mlp.down_proj": layer.mlp.down_proj,
    }

    scaling = config.lora_alpha / config.lora_r
    delta_telemetry: list[dict[str, float | str | int]] = []

    with torch.no_grad():
        for name, lora_layer in target_dict.items():
            # A has shape (r, d_in); B has shape (d_out, r)
            lora_A = lora_layer.lora_A["default"].weight.detach().to("cpu", torch.float32)
            lora_B = lora_layer.lora_B["default"].weight.detach().to("cpu", torch.float32)
            base_W = lora_layer.base_layer.weight.detach().to("cpu", torch.float32)

            # Compute exact low-rank update matrix Delta_W = (alpha / r) * (B @ A)
            delta_W = scaling * (lora_B @ lora_A)

            frob_delta = torch.linalg.norm(delta_W, ord="fro").item()
            frob_base = torch.linalg.norm(base_W, ord="fro").item()
            relative_update_pct = (frob_delta / frob_base) * 100.0

            # Compute singular values of the r x d_in / d_out x r product via QR/SVD
            # Since rank(B @ A) <= r, only top `r` singular values can be non-zero!
            s_delta = torch.linalg.svdvals(delta_W)
            nonzero_rank = int(torch.sum(s_delta > 1e-5).item())

            print(
                f"{name:<20} | {frob_delta:>14.5f} | {relative_update_pct:>19.4f}% | "
                f"{s_delta[0].item():>16.5f} | {nonzero_rank:>6d} (<= r={config.lora_r})"
            )
            delta_telemetry.append(
                {
                    "module": name,
                    "frob_delta": round(frob_delta, 5),
                    "relative_update_pct": round(relative_update_pct, 4),
                    "sigma_1_delta": round(s_delta[0].item(), 5),
                    "nonzero_rank": nonzero_rank,
                }
            )

    return peft_model, delta_telemetry, initial_loss, final_loss


# ==============================================================================
# PART 4: GENERATE MODEL CARD & PUSH TO HUGGING FACE HUB (`garg-achal`)
# ==============================================================================


def generate_model_card(
    config: LabConfig,
    svd_telemetry: list[dict[str, float | str | tuple[int, int]]],
    delta_telemetry: list[dict[str, float | str | int]],
    initial_loss: float,
    final_loss: float,
) -> str:
    """Create a portfolio-grade Hugging Face Model Card with live SVD & LoRA telemetry."""
    svd_rows = "\n".join(
        f"| `{row['module']}` | `{row['shape']}` | `{row['sigma_1']}` | `{row['top8_energy_pct']}%` | `{row['effective_rank']} / {row['full_rank']}` |"
        for row in svd_telemetry
    )
    delta_rows = "\n".join(
        f"| `{row['module']}` | `{row['frob_delta']}` | `{row['relative_update_pct']}%` | `{row['sigma_1_delta']}` | `{row['nonzero_rank']}` |"
        for row in delta_telemetry
    )

    return f"""---
library_name: peft
base_model: {config.model_id}
tags:
  - lora
  - trl
  - sft
  - finops
  - svd
  - weight-surgery
  - apple-silicon-mps
  - applied-ai-foundry
license: apache-2.0
---

# {config.hub_repo_id.split('/')[-1]}: Weight-Level SVD Dissection & Enterprise FinOps LoRA Adapter

Fine-tuned **PEFT LoRA ($r={config.lora_r}, \\alpha={config.lora_alpha}$)** adapter for **[`{config.model_id}`](https://huggingface.co/{config.model_id})** built as part of the **Applied AI Foundry (`v4.0`)** Sprint (**[`curons.org`](https://curons.org)** | **[`@garg-achal`](https://huggingface.co/garg-achal)**).

This checkpoint trains `{config.model_id}` to produce deterministic **`[FINOPS_AUDIT_REPORT]`** diagnostics for enterprise GPU sizing, KV-cache bottlenecks, and `vLLM` serving optimizations—accompanied by **Singular Value Decomposition (SVD)** spectrum telemetry of both the base weights $W$ and the low-rank adapter update $\\Delta W = \\frac{{\\alpha}}{{r}} B \\cdot A$.

---

## 1. Base Model Weight Spectrum (Layer {config.inspect_layer_idx} SVD Analysis)

| Projection Module | Weight Shape $(d_{{\\text{{out}}}}, d_{{\\text{{in}}}})$ | Top Singular Value ($\\sigma_1$) | Top-8 Spectral Energy % | Effective Shannon Rank / Full Rank |
| :--- | :--- | :--- | :--- | :--- |
{svd_rows}

---

## 2. LoRA Adapter Trajectory ($\\Delta W = \\frac{{\\alpha}}{{r}} B \\cdot A$ at Layer {config.inspect_layer_idx})

- **Training Hardware**: Apple M4 (24GB Unified Memory, PyTorch `mps` `bfloat16`)
- **Framework**: Hugging Face `transformers` v5 + `peft` + `trl.SFTTrainer`
- **Convergence**: Initial SFT Loss `{initial_loss:.4f}` $\\rightarrow$ Final SFT Loss `{final_loss:.4f}` ({config.max_steps} steps)

| Target Module | $\\|\\Delta W\\|_F$ (Frobenius Norm) | Relative Update $\\|\\Delta W\\|_F / \\|W\\|_F$ | $\\sigma_1(\\Delta W)$ | Verified Rank ($\\le r={config.lora_r}$) |
| :--- | :--- | :--- | :--- | :--- |
{delta_rows}

---

## 3. Quickstart Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "{config.model_id}"
adapter_id = "{config.hub_repo_id}"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base_model, adapter_id)
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the trained LoRA adapter and SVD Model Card to Hugging Face Hub (garg-achal).",
    )
    parser.add_argument(
        "--svd-only",
        action="store_true",
        help="Run only the Layer 18 weight dissection and SVD spectrum analysis, then exit.",
    )
    args = parser.parse_args()

    config = LabConfig()
    logger.info(
        "Starting Week 1 SVD & LoRA Lab on device='%s' with model='%s'",
        config.device,
        config.model_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=config.dtype,
        device_map=config.device,
        attn_implementation="sdpa",
    )

    # 1. Dissect base transformer weights & compute SVD spectrum
    svd_telemetry = dissect_transformer_weights_and_svd(model, config.inspect_layer_idx)

    if args.svd_only:
        logger.info(
            "Completed SVD spectrum analysis for layer %d (--svd-only flag enabled). Exiting cleanly.",
            config.inspect_layer_idx,
        )
        return

    # 2. Train LoRA adapter via TRL SFTTrainer & analyze Delta_W = (alpha/r) * B @ A
    peft_model, delta_telemetry, initial_loss, final_loss = (
        run_lora_sft_and_inspect_delta_w(model, tokenizer, config)
    )

    # 3. Verify live generation with the trained FinOps LoRA adapter
    print("\n" + "=" * 86)
    print("4. POST-TRAINING INFERENCE VERIFICATION ([FINOPS_AUDIT_REPORT] FORMAT)")
    print("=" * 86)
    peft_model.eval()
    test_messages = [
        {
            "role": "system",
            "content": "You are the Lead AI Systems & FinOps Architect at curons.org. Always respond using the structured [FINOPS_AUDIT_REPORT] format with Root Cause, Remediation, and Projected ROI.",
        },
        {
            "role": "user",
            "content": "We have 12 enterprise product teams each wanting a custom fine-tuned 4B model. Do we need 12 separate GPU instances?",
        },
    ]
    inputs = tokenizer.apply_chat_template(
        test_messages,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(config.device)

    with torch.inference_mode():
        out_ids = peft_model.generate(
            **inputs,
            max_new_tokens=180,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_text = tokenizer.decode(
        out_ids[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    print(gen_text.strip())
    print("=" * 86)

    # 4. Write Model Card locally and optionally push to Hugging Face Hub
    card_md = generate_model_card(
        config, svd_telemetry, delta_telemetry, initial_loss, final_loss
    )
    readme_path = f"{config.output_dir}/README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(card_md)
    logger.info("Saved SVD & LoRA Model Card to %s", readme_path)

    if args.push_to_hub:
        logger.info("Pushing LoRA adapter & SVD Model Card to https://huggingface.co/%s ...", config.hub_repo_id)
        peft_model.push_to_hub(config.hub_repo_id)
        tokenizer.push_to_hub(config.hub_repo_id)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=config.hub_repo_id,
            repo_type="model",
            commit_message="docs: add Week 1 SVD weight spectrum & LoRA Delta_W trajectory telemetry",
        )
        logger.info("Successfully published to https://huggingface.co/%s !", config.hub_repo_id)


if __name__ == "__main__":
    main()
