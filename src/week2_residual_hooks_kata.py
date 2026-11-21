"""
Week 2, Lab 1 (Prototype 1 Kata — Write-First / Anti-Vibe-Coding Gate):
Transformer Residual Stream Hooks, Contrastive Steering Vectors & Linear Probe Clamping

Target Model : Qwen/Qwen3-4B (36 layers, d_model=2560, Apple M4 `mps` / bfloat16)
Tournament   : Prototype 1 (Runtime Activation Steering & Production Probe Classifiers)

ANTI-VIBE-CODING PROTOCOL:
1. Hand-write the 3 core functions below (`make_activation_capture_hook`,
   `compute_contrastive_steering_vector`, `make_activation_steering_hook`)
   without AI autocomplete (~25-30 lines total).
2. Run the deterministic 4/4 verification suite locally:
       .venv/bin/python src/week2_residual_hooks_kata.py
3. Once 4/4 tests pass and you pass the Explain-Back Gate, unload LM Studio
   (`lms unload --all`) and benchmark Layer 18 hooks live on Qwen/Qwen3-4B (M4 MPS):
       .venv/bin/python src/week2_residual_hooks_kata.py --live-mps
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# PART 1: CORE KATAS (HAND-WRITE THESE 3 FUNCTIONS WITHOUT AI AUTOCOMPLETE)
# ==============================================================================


def make_activation_capture_hook(
    cache: dict[str, torch.Tensor],
    key: str,
) -> Callable[[nn.Module, tuple[Any, ...], Any], None]:
    """Return a PyTorch forward hook that captures the last-token residual activation.

    PyTorch invokes forward hooks registered via `module.register_forward_hook(hook)`
    with `(module, inputs, output)`. Depending on the Hugging Face `transformers`
    version and layer configuration, a decoder block's `output` may be either:
      - a `torch.Tensor` of shape `[batch_size, seq_len, d_model]`, OR
      - a `tuple` whose first element `output[0]` is that `hidden_states` tensor.

    Requirements:
      1. Extract `hidden_states = output[0] if isinstance(output, tuple) else output`.
      2. Slice the final token position `hidden_states[:, -1, :]` (shape `[batch_size, d_model]`).
      3. Detach from the autograd graph, move to `"cpu"`, and cast to `torch.float32`.
      4. Store the resulting tensor in `cache[key]`.
      5. Return `None` so the forward pass continues unmodified.
    """
    # TODO (Hand-Write ~8-10 lines): Define and return `hook_fn(module, inputs, output)`.
    raise NotImplementedError(
        "Hand-write make_activation_capture_hook() to unlock the kata gate!"
    )


def compute_contrastive_steering_vector(
    truthful_acts: torch.Tensor,
    hallucinated_acts: torch.Tensor,
) -> torch.Tensor:
    """Compute a unit-norm Contrastive Activation Addition (CAA) steering vector.

    Given captured residual stream activations for truthful (`H^+` of shape `[N, d_model]`)
    and hallucinated (`H^-` of shape `[M, d_model]`) prompts:
        delta   = mean(H^+, dim=0) - mean(H^-, dim=0)
        v_truth = delta / (||delta||_2 + 1e-12)

    Args:
        truthful_acts: Float tensor of shape `(num_truthful, d_model)`.
        hallucinated_acts: Float tensor of shape `(num_hallucinated, d_model)`.

    Returns:
        1D unit-norm tensor `v_truth` of shape `(d_model,)` in `torch.float32`.
    """
    # TODO (Hand-Write ~5-7 lines): Compute difference of means and L2-normalize.
    raise NotImplementedError(
        "Hand-write compute_contrastive_steering_vector() to unlock the kata gate!"
    )


def make_activation_steering_hook(
    steering_vector: torch.Tensor,
    alpha: float,
) -> Callable[[nn.Module, tuple[Any, ...], Any], Any]:
    """Return a PyTorch forward hook that clamps/steers the residual stream in flight.

    When registered on a transformer decoder layer (`model.model.layers[18]`),
    the hook shifts the layer's output hidden states along `steering_vector`:
        steered = hidden_states + alpha * v

    Requirements:
      1. Extract `hidden_states = output[0] if isinstance(output, tuple) else output`.
      2. Cast/move `steering_vector` to match `hidden_states.device` and `hidden_states.dtype`.
      3. Compute `steered = hidden_states + alpha * v`.
      4. If `output` was a tuple, return `(steered,) + output[1:]` so trailing states
         (such as cached KV tensors or attention weights) pass through intact;
         otherwise return `steered`.
    """
    # TODO (Hand-Write ~8-10 lines): Define and return `hook_fn(module, inputs, output)`.
    raise NotImplementedError(
        "Hand-write make_activation_steering_hook() to unlock the kata gate!"
    )


# ==============================================================================
# PART 2: SCAFFOLDED TEST HARNESS (DO NOT MODIFY BELOW)
# ==============================================================================


class _SyntheticTensorBlock(nn.Module):
    """Simulates a transformers v5 decoder layer returning a single hidden_states Tensor."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.proj(x)


class _SyntheticTupleBlock(nn.Module):
    """Simulates a decoder layer returning `(hidden_states, kv_state)` as a tuple."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, kv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = x + 0.1 * self.proj(x)
        return (out, kv_state)


class _SyntheticMiniTransformer(nn.Module):
    """Minimal 3-layer residual stack (d_model=2560) with a linear lm_head."""

    def __init__(self, d_model: int = 2560, vocab_size: int = 32) -> None:
        super().__init__()
        self.layer_0 = _SyntheticTensorBlock(d_model)
        self.layer_1 = _SyntheticTupleBlock(d_model)  # Middle target layer (analogous to Layer 18)
        self.layer_2 = _SyntheticTensorBlock(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, x: torch.Tensor, kv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.layer_0(x)
        h, kv_out = self.layer_1(h, kv_state)
        h = self.layer_2(h)
        logits = self.lm_head(h[:, -1, :])
        return logits, kv_out


def run_verification_suite() -> None:
    print("=" * 78)
    print("WEEK 2 LAB 1: RESIDUAL HOOKS & ACTIVATION STEERING VERIFICATION SUITE")
    print("=" * 78)

    torch.manual_seed(42)
    d_model = 2560  # Exact hidden dimension of Qwen/Qwen3-4B

    # --------------------------------------------------------------------------
    # Test 1/4: Capture Hook (Tuple + Tensor outputs, detach/cpu/fp32, cleanup)
    # --------------------------------------------------------------------------
    tuple_block = _SyntheticTupleBlock(d_model).to(dtype=torch.bfloat16)
    tensor_block = _SyntheticTensorBlock(d_model).to(dtype=torch.bfloat16)
    cache: dict[str, torch.Tensor] = {}

    x_in = torch.randn(2, 6, d_model, dtype=torch.bfloat16, requires_grad=True)
    dummy_kv = torch.randn(2, 4, 6, 64, dtype=torch.bfloat16)

    h1 = tuple_block.register_forward_hook(
        make_activation_capture_hook(cache, "layer_1_tuple")
    )
    h2 = tensor_block.register_forward_hook(
        make_activation_capture_hook(cache, "layer_2_tensor")
    )
    try:
        out_tuple, kv_after = tuple_block(x_in, dummy_kv)
        out_tensor = tensor_block(out_tuple)
    finally:
        h1.remove()
        h2.remove()

    cap_tuple = cache["layer_1_tuple"]
    cap_tensor = cache["layer_2_tensor"]
    assert cap_tuple.shape == (2, d_model), (
        f"Expected captured shape (2, {d_model}), got {cap_tuple.shape}"
    )
    assert cap_tuple.dtype == torch.float32 and cap_tensor.dtype == torch.float32, (
        f"Captured activations must be float32, got {cap_tuple.dtype} and {cap_tensor.dtype}"
    )
    assert cap_tuple.device.type == "cpu" and cap_tensor.device.type == "cpu", (
        "Captured activations must live on CPU to prevent GPU memory buildup."
    )
    assert not cap_tuple.requires_grad and not cap_tensor.requires_grad, (
        "Captured activations must be detached (.detach()) from autograd."
    )
    expected_last_tuple = out_tuple[:, -1, :].detach().to(device="cpu", dtype=torch.float32)
    expected_last_tensor = out_tensor[:, -1, :].detach().to(device="cpu", dtype=torch.float32)
    assert torch.allclose(cap_tuple, expected_last_tuple, atol=1e-6), (
        "Tuple-layer captured tensor does not match final token hidden state."
    )
    assert torch.allclose(cap_tensor, expected_last_tensor, atol=1e-6), (
        "Tensor-layer captured tensor does not match final token hidden state."
    )
    assert torch.equal(kv_after, dummy_kv), "Capture hook mutated trailing tuple KV state."

    # Verify handle.remove() cleanly stops further captures
    cache.clear()
    with torch.no_grad():
        _ = tuple_block(x_in.detach(), dummy_kv)
        _ = tensor_block(x_in.detach())
    assert len(cache) == 0, "Removed hooks still wrote into cache!"
    print("[PASS] 1/4 make_activation_capture_hook() captures & detaches tuple/tensor streams.")

    # --------------------------------------------------------------------------
    # Test 2/4: Contrastive Steering Vector (CAA unit norm & cluster alignment)
    # --------------------------------------------------------------------------
    u_true = torch.randn(d_model, dtype=torch.float32)
    u_true = u_true / u_true.norm(p=2)

    truthful_acts = 3.0 * u_true + 0.15 * torch.randn(64, d_model, dtype=torch.float32)
    hallucinated_acts = -3.0 * u_true + 0.15 * torch.randn(48, d_model, dtype=torch.float32)

    v_truth = compute_contrastive_steering_vector(truthful_acts, hallucinated_acts)
    assert v_truth.shape == (d_model,), f"Expected shape ({d_model},), got {v_truth.shape}"
    v_norm = v_truth.norm(p=2).item()
    assert abs(v_norm - 1.0) < 1e-5, f"Expected unit L2 norm 1.0, got {v_norm:.6f}"

    cos_pos = F.cosine_similarity(truthful_acts.mean(dim=0), v_truth, dim=0).item()
    cos_neg = F.cosine_similarity(hallucinated_acts.mean(dim=0), v_truth, dim=0).item()
    cos_oracle = F.cosine_similarity(v_truth, u_true, dim=0).item()
    assert cos_pos > 0.95 and cos_neg < -0.95 and cos_oracle > 0.98, (
        f"Steering vector poorly aligned: cos_pos={cos_pos:.3f}, cos_neg={cos_neg:.3f}, oracle={cos_oracle:.3f}"
    )
    print(
        f"[PASS] 2/4 compute_contrastive_steering_vector() unit norm={v_norm:.6f} "
        f"(oracle cosine={cos_oracle:.4f})."
    )

    # --------------------------------------------------------------------------
    # Test 3/4: Runtime Steering Hook (Exact shift, bfloat16 dtype, no mutation)
    # --------------------------------------------------------------------------
    alpha = 4.0
    x_test = torch.randn(2, 5, d_model, dtype=torch.bfloat16)
    w_tuple_before = tuple_block.proj.weight.detach().clone()
    w_tensor_before = tensor_block.proj.weight.detach().clone()

    with torch.no_grad():
        base_tuple_out, base_kv = tuple_block(x_test, dummy_kv)
        base_tensor_out = tensor_block(x_test)

        s_h1 = tuple_block.register_forward_hook(
            make_activation_steering_hook(v_truth, alpha=alpha)
        )
        s_h2 = tensor_block.register_forward_hook(
            make_activation_steering_hook(v_truth, alpha=alpha)
        )
        try:
            steered_tuple_out, steered_kv = tuple_block(x_test, dummy_kv)
            steered_tensor_out = tensor_block(x_test)
        finally:
            s_h1.remove()
            s_h2.remove()

    assert steered_tuple_out.dtype == torch.bfloat16 and steered_tensor_out.dtype == torch.bfloat16, (
        "Steering hook changed hidden_states dtype!"
    )
    delta_bf16 = alpha * v_truth.to(dtype=torch.bfloat16)
    assert torch.allclose(
        steered_tuple_out.float(), (base_tuple_out + delta_bf16).float(), atol=5e-2, rtol=1e-2
    ), "Tuple steering shift mismatch."
    assert torch.allclose(
        steered_tensor_out.float(), (base_tensor_out + delta_bf16).float(), atol=5e-2, rtol=1e-2
    ), "Tensor steering shift mismatch."
    assert torch.equal(steered_kv, base_kv), "Steering hook altered trailing tuple KV state."
    assert torch.equal(tuple_block.proj.weight, w_tuple_before) and torch.equal(
        tensor_block.proj.weight, w_tensor_before
    ), "Steering hook unexpectedly mutated layer weights."
    print("[PASS] 3/4 make_activation_steering_hook() shifts tuple/tensor streams cleanly.")

    # --------------------------------------------------------------------------
    # Test 4/4: Linear Probe + In-Flight Clamping & <2ms Latency Overhead Check
    # --------------------------------------------------------------------------
    model = _SyntheticMiniTransformer(d_model=d_model, vocab_size=32).eval()
    with torch.no_grad():
        # Keep layer projections small so residual stream direction remains clean
        for blk in (model.layer_0, model.layer_1, model.layer_2):
            blk.proj.weight.mul_(0.02)
        model.lm_head.weight.zero_()
        model.lm_head.weight[0] = 2.0 * v_truth  # Token 0: grounded/verified JSON output
        model.lm_head.weight[1] = -2.0 * v_truth  # Token 1: hallucinated savings claim

    # Lightweight linear hallucination probe: P(halluc | h_mid) = sigmoid(w^T h + b)
    probe_w = -4.0 * v_truth
    probe_b = 0.0

    # Input anchored in the hallucination basin (-1.5 * v_truth)
    x_halluc = (-1.5 * v_truth).view(1, 1, d_model).repeat(1, 4, 1)
    kv_zero = torch.zeros(1, 4, 4, 64)
    probe_cache: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        cap_handle = model.layer_1.register_forward_hook(
            make_activation_capture_hook(probe_cache, "mid")
        )
        try:
            unsteered_logits, _ = model(x_halluc, kv_zero)
        finally:
            cap_handle.remove()

        p_halluc = torch.sigmoid(
            torch.dot(probe_cache["mid"].squeeze(0), probe_w) + probe_b
        ).item()
        assert p_halluc > 0.90, f"Probe failed to flag hallucination state (p={p_halluc:.3f})"
        assert unsteered_logits[0, 1] > unsteered_logits[0, 0], (
            "Unsteered model did not favor hallucinated token 1 in baseline setup."
        )

        # Probe triggered -> attach runtime steering hook in-flight
        steer_handle = model.layer_1.register_forward_hook(
            make_activation_steering_hook(v_truth, alpha=3.5)
        )
        try:
            steered_logits, _ = model(x_halluc, kv_zero)
        finally:
            steer_handle.remove()

        assert steered_logits[0, 0] > steered_logits[0, 1], (
            "Steering failed to flip prediction from hallucinated token 1 to verified token 0!"
        )

        # Benchmark per-forward-pass hook + probe overhead (target < 2.0 ms)
        iters = 50
        for _ in range(10):  # Warmup
            _ = model(x_halluc, kv_zero)

        t0 = time.perf_counter()
        for _ in range(iters):
            _ = model(x_halluc, kv_zero)
        base_ms = ((time.perf_counter() - t0) / iters) * 1000.0

        c_h = model.layer_1.register_forward_hook(
            make_activation_capture_hook(probe_cache, "mid")
        )
        s_h = model.layer_1.register_forward_hook(
            make_activation_steering_hook(v_truth, alpha=3.5)
        )
        try:
            for _ in range(10):  # Warmup
                _, _ = model(x_halluc, kv_zero)
                _ = torch.sigmoid(torch.dot(probe_cache["mid"].squeeze(0), probe_w) + probe_b)

            t1 = time.perf_counter()
            for _ in range(iters):
                _, _ = model(x_halluc, kv_zero)
                _ = torch.sigmoid(torch.dot(probe_cache["mid"].squeeze(0), probe_w) + probe_b)
            hooked_ms = ((time.perf_counter() - t1) / iters) * 1000.0
        finally:
            c_h.remove()
            s_h.remove()

    overhead_ms = max(0.0, hooked_ms - base_ms)
    assert overhead_ms < 2.0, (
        f"Hook + probe overhead {overhead_ms:.3f} ms exceeded 2.0 ms FinOps SLA."
    )
    print(
        f"[PASS] 4/4 Probe triggered (P_halluc={p_halluc:.3f}) -> flipped logit margin "
        f"| Hook+Probe Overhead: {overhead_ms:.3f} ms (<2.0 ms SLA)"
    )
    print("=" * 78)
    print("KATA COMPLETE! Ready for Explain-Back Gate & `--live-mps` Qwen3-4B Run.")
    print("=" * 78)


# ==============================================================================
# PART 3: LIVE QWEN/QWEN3-4B LAYER 18 HOOK & LATENCY BENCHMARK (APPLE M4 MPS)
# ==============================================================================


def _sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def run_live_qwen3_mps_demo(model_id: str = "Qwen/Qwen3-4B", layer_idx: int = 18) -> None:
    """Attach your hand-written hooks to Qwen3-4B Layer 18 on Apple M4 MPS."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    print("\n" + "=" * 78)
    print(f"LIVE MODEL DEMO: {model_id} | Layer {layer_idx} Residual Stream Hooks")
    print("=" * 78)
    print(
        "M4 Memory Guardrail: If LM Studio is running Qwen3-Coder-30B-A3B (~14.5 GB),\n"
        "run `lms unload --all` first to prevent Unified Memory swap."
    )

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32

    print(f"Loading tokenizer & model on device='{device}' (dtype={dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",
    )
    model.eval()

    target_layer = model.model.layers[layer_idx]
    print(
        f"Target block: model.model.layers[{layer_idx}] of {model.config.num_hidden_layers} "
        f"(d_model={model.config.hidden_size})"
    )

    # Contrastive FinOps pairs: grounded engineering facts vs. hallucinated claims
    truthful_prompts = [
        "Explain why vLLM PagedAttention cuts KV-cache fragmentation on an A10G GPU.",
        "Explain why causal decoder-only batching requires padding_side='left'.",
        "Summarize how rank-8 LoRA adapters share a single base model via --enable-lora.",
        "State how W4A16 quantization reduces 7B model weight VRAM from 14 GB to ~4.2 GB.",
    ]
    hallucinated_prompts = [
        "Confirm that a 70B FP16 dense LLM fits cleanly inside a single 8 GB VRAM GPU with 32k context.",
        "Explain why decoder-only models require padding_side='right' so the -1 token reads pad IDs.",
        "Prove that PyTorch forward hooks add 600ms of network latency per generated token.",
        "Confirm that LoRA rank-8 updates modify all 4 billion base weights in-place during SFT.",
    ]

    def _collect_layer_acts(prompts: list[str]) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        cache: dict[str, torch.Tensor] = {}
        handle = target_layer.register_forward_hook(
            make_activation_capture_hook(cache, "residual")
        )
        try:
            with torch.inference_mode():
                for text in prompts:
                    msgs = [{"role": "user", "content": text}]
                    inputs = tokenizer.apply_chat_template(
                        msgs,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_tensors="pt",
                        return_dict=True,
                    ).to(device)
                    _ = model(**inputs, use_cache=False)
                    vectors.append(cache["residual"].squeeze(0))
        finally:
            handle.remove()
        return torch.stack(vectors, dim=0)

    truthful_acts = _collect_layer_acts(truthful_prompts)
    hallucinated_acts = _collect_layer_acts(hallucinated_prompts)
    v_truth = compute_contrastive_steering_vector(truthful_acts, hallucinated_acts)

    truth_proj = (truthful_acts @ v_truth).mean().item()
    halluc_proj = (hallucinated_acts @ v_truth).mean().item()
    print(
        f"Layer {layer_idx} CAA Vector Computed: shape={tuple(v_truth.shape)}, "
        f"norm={v_truth.norm().item():.5f} | Mean Projection: "
        f"truthful={truth_proj:+.3f} vs hallucinated={halluc_proj:+.3f}"
    )

    # Benchmark baseline vs. hooked forward pass overhead on M4 MPS
    probe_prompt = "Can a 70B FP16 model run inside 24 GB VRAM without quantization? Answer in one sentence."
    probe_inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": probe_prompt}],
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    bench_iters = 12
    hook_cache: dict[str, torch.Tensor] = {}

    with torch.inference_mode():
        for _ in range(3):
            _ = model(**probe_inputs, use_cache=False)
        _sync_device(device)

        t0 = time.perf_counter()
        for _ in range(bench_iters):
            _ = model(**probe_inputs, use_cache=False)
        _sync_device(device)
        base_ms = ((time.perf_counter() - t0) / bench_iters) * 1000.0

        c_handle = target_layer.register_forward_hook(
            make_activation_capture_hook(hook_cache, "live")
        )
        s_handle = target_layer.register_forward_hook(
            make_activation_steering_hook(v_truth, alpha=2.0)
        )
        try:
            for _ in range(3):
                _ = model(**probe_inputs, use_cache=False)
            _sync_device(device)

            t1 = time.perf_counter()
            for _ in range(bench_iters):
                _ = model(**probe_inputs, use_cache=False)
            _sync_device(device)
            hooked_ms = ((time.perf_counter() - t1) / bench_iters) * 1000.0
        finally:
            c_handle.remove()
            s_handle.remove()

    print(
        f"Prefill Latency ({device}): Baseline={base_ms:.2f} ms | "
        f"Hooked (Capture+Steer)={hooked_ms:.2f} ms | "
        f"Hook Overhead={max(0.0, hooked_ms - base_ms):.2f} ms"
    )

    # Verify DynamicCache compatibility during multi-token generation with steering active
    with torch.inference_mode():
        kv_cache = DynamicCache()
        gen_handle = target_layer.register_forward_hook(
            make_activation_steering_hook(v_truth, alpha=2.0)
        )
        try:
            out_ids = model.generate(
                **probe_inputs,
                past_key_values=kv_cache,
                max_new_tokens=48,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        finally:
            gen_handle.remove()

    generated = tokenizer.decode(
        out_ids[0, probe_inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    print(
        f"\nSteered Generation (alpha=2.0, KV-Cache Len={kv_cache.get_seq_length()}):\n"
        f"{generated.strip()}"
    )
    if device == "mps":
        torch.mps.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Week 2 Lab 1: PyTorch Residual Stream Hooks & Steering Kata"
    )
    parser.add_argument(
        "--live-mps",
        action="store_true",
        help="After passing the 4/4 kata suite, run Layer 18 hooks live on Qwen/Qwen3-4B.",
    )
    args = parser.parse_args()

    run_verification_suite()
    if args.live_mps:
        run_live_qwen3_mps_demo()


if __name__ == "__main__":
    main()
