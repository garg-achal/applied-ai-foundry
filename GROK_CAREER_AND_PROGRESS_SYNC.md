# Grok Bot Context: `Applied AI Foundry` (v4.0) — Weight-Level Systems, MoE Surgery, Activation Steering & AI Venture Sync

> **SYSTEM INSTRUCTIONS FOR GROK BOT:**  
> Use this living document as your single source of truth to:  
> 1. **Senior/Staff Job Search & Role Matching (8–12+ YOE Band — Never Junior Roles)**: Search for high-impact roles in India (Delhi NCR, Bengaluru, Hyderabad, Pune) and Global Remote, prioritizing the candidate's top tracks:  
>    - **Primary Track 1 (Infra / Systems)**: **Senior / Staff Inference, MLOps & AI Platform Engineer (LLM Serving / GenAI Infra)**  
>    - **Primary Track 2 (Leadership & Delivery)**: **Staff / Principal Technical Program Manager — GenAI / AI Platform / LLMOps**  
>    - **Primary Track 3 (Product & Strategy)**: **Senior / Principal AI Product Manager (AI PM — LLM Platform & FinOps)**  
>    - **Secondary Track 4 (Model Customization)**: **Lead / Senior Applied AI & Post-Training Engineer (Fine-Tuning, Evals & Agents)**  
>    - **Secondary Track 5 (Customer Architecture)**: **Forward Deployed Engineer (FDE) / Principal AI Solutions Architect**  
> 2. **Founder & Venture Positioning (`curons.org`)**:  
>    - Portray the candidate as an **Independent AI Founder and 10–13 Year Software Systems Leader** building at the weight, neuron, and activation level (TRL alignment, MoE pruning, runtime activation steering, and vLLM inference economics).  
>    - Core B2B Commercial Offer: A fixed-price **2-Week LLM FinOps & Weight Optimization Audit (\$3,000–\$5,000)** that helps enterprise engineering teams cut GPU inference bills by 60–85% while guaranteeing sub-second P95 latency and >99% schema adherence.  
> 3. **Time Commitment & Pacing Constraints**:  
>    - The candidate is on a **strict 42-hour weekly budget** (7 hours/day Monday through Saturday, with Sundays reserved for rest, synthesis, and 90-minute VAE/GAN review).  
> 4. **Mock Interview & Technical Scenario Drills**: Quiz the candidate using the weekly scenarios in Section 4.

---

## 1. Candidate Profile & Positioning Playbook

- **Experience**: 10–13 Years in Software Systems, Cloud Infrastructure, AI Platform Delivery & Technical Leadership (ex-Uniphore, ex-Bharti Airtel).  
- **Venture Entity**: `curons.org` (AI Systems Foundry & FinOps Advisory).  
- **GitHub Foundry Repo**: [`https://github.com/garg-achal/applied-ai-foundry`](https://github.com/garg-achal/applied-ai-foundry) (`garg-achal/applied-ai-foundry` — Reference architecture for QLoRA, MoE Pruning, Activation Steering, and vLLM Multi-LoRA).  
- **Hugging Face Hub**: [`https://huggingface.co/imhlg`](https://huggingface.co/imhlg) (`imhlg`).  
- **Sprint Horizon**: **v4.0 `Applied AI Foundry` Sprint** (Nov 1 – Dec 13, 2026 Core 6-Week Sprint @ 42 hrs/wk = 252 Hours; Phase 1 B2B Pilots & Senior Applications launch Nov 30; Weeks 7–10 Advanced Horizon for Evolutionary Merging & Triton Kernels).

### Three Tailored Positioning Variants for Outreach & Applications

1. **Variant A (Staff/Principal AI Platform TPM & AI Product Leader)**:  
   - *Headline*: `Staff Technical Program Manager & AI Platform Architect | GenAI Infra, High-Throughput vLLM Serving, Eval CI/CD & GPU FinOps`  
   - *Core Hook*: Combines 10+ years of cross-functional platform delivery and enterprise cloud TCO ownership with hands-on post-training and inference platform builds.  
2. **Variant B (Senior/Staff AI Platform & Inference Systems Engineer)**:  
   - *Headline*: `Senior AI Systems & Inference Platform Engineer | Post-Training (LoRA/QLoRA), MoE Pruning, Activation Steering & vLLM Serving`  
   - *Core Hook*: Distributed systems builder specializing in PyTorch forward hooks, QLoRA adaptation via TRL, MoE expert pruning, and low-latency continuous batching in vLLM.  
3. **Variant C (AI Founder / Technical Advisory — `curons.org`)**:  
   - *Headline*: `Founder @ curons.org | Slashing Enterprise LLM Cloud Spend via MoE Surgery, Weight Adaptation & vLLM High-Throughput Serving`  
   - *Core Hook*: Helping CTOs and VPs of Engineering replace costly frontier APIs with self-hosted, fine-tuned open-weight models at 80%+ lower blended token costs.

### Active Hardware & Production Stack
- **Local Stack**: Apple M4 (24GB Unified Memory), PyTorch `mps` (`torch 2.14.1`), Apple `mlx-lm`, `llama.cpp` (Metal), LM Studio (`Qwen3-Coder-30B-A3B` `Q3_K_L`).  
- **Cloud Stack**: NVIDIA L4 / A10G GPUs (Ada SM 8.9 / Ampere) on Google Colab Pro & AWS EC2 (`g5.xlarge`), `bitsandbytes` NF4, `llm-compressor`, `vLLM` with PagedAttention and continuous batching.  
- **AI Platform Stack**: Hugging Face `transformers` v5, `datasets`, `peft`, `trl` v1+, `Outlines` constrained decoding, `FastMCP` (Model Context Protocol), `promptfoo`, `DeepEval`, `RAGAS`, OpenTelemetry `gen_ai.*` tracing (`Arize Phoenix`), async `FastAPI` AI Gateway, Docker, GCP Cloud Run, and Hugging Face Spaces.
- **Developer Environment**: **VS Code IDE** + **Antigravity Desktop App** (integrated right-side terminal execution pane) + **Claude Code CLI (`claude` / `CLAUDE.md`)**.

---

## 2. Live 6-Week Sprint Matrix (Nov 1 – Dec 13, 2026 | 42 hrs/wk)

| Week & Dates | Module Focus | Status | Key Technologies & Architecture Keywords |
| :--- | :--- | :--- | :--- |
| **Week 1** *(Nov 1–Nov 8)* | **Tokenization, Weight Matrices ($W_q, W_{mlp}$) & SVD, TRL QLoRA & W4A16** | 🟢 **100% Complete (Local M4 + Colab L4 NF4 QLoRA & W4A16)** | Byte-Level BPE (`minbpe` kata), Left-Padding masks, SVD rank analysis, `TRL` `SFTTrainer`, `bitsandbytes` NF4 QLoRA, `llm-compressor` GPTQ W4A16, HF Hub ([`imhlg/qwen3-4b-finops-lora-week1`](https://huggingface.co/imhlg/qwen3-4b-finops-lora-week1), [`imhlg/qwen3-4b-finops-w4a16-vllm`](https://huggingface.co/imhlg/qwen3-4b-finops-w4a16-vllm)) |
| **Week 2** *(Nov 9–Nov 15)* | **Residual Streams, Runtime Activation Steering & Probe Classifiers** *(Prototype 1)* | 🟢 **Colab L4 Prototype 1 Shipped (Local 25-line Kata in progress)** | PyTorch forward hooks (`register_forward_hook`), `Outlines` guided JSON decoding, 1k-prompt sweep ([`datasets/imhlg/finops-residual-activations-1k`](https://huggingface.co/datasets/imhlg/finops-residual-activations-1k)), `20.21 KB` Layer 18 probe + CAA $\vec{v}_{\text{truth}}$ (`0.924 ms` overhead, [`imhlg/qwen3-4b-activation-probes-week2`](https://huggingface.co/imhlg/qwen3-4b-activation-probes-week2)) |
| **Week 3** *(Nov 16–Nov 22)* | **MoE Router Profiling, Structured Expert Pruning & Distillation** *(Prototype 2)* | ⚪ Planned | `OLMoE-1B-7B` / `Qwen2.5-MoE`, Router softmax gate logits profiling, Mean Squared Activation Norm (MSAN), physical expert tensor excision, TRL distillation recovery |
| **Week 4** *(Nov 23–Nov 29)* | **Machine Unlearning (NPO), vLLM Serving & Tournament Decision** *(Prototype 3)* | ⚪ Planned | Negative Preference Optimization (`TRL`), rank-one weight editing (ROME), `vLLM` PagedAttention & continuous batching, async load testing (TTFT/ITL), **Sunday Tournament Lock** |
| **Week 5** *(Nov 30–Dec 6)* | **Productize Winning Thesis, Open-Source CLI & B2B Pilot Launch** | ⚪ Planned | Clean repo packaging (`applied-ai-foundry`), GitHub Actions CI eval gates, FinOps unit-cost & latency calculator, `curons.org` landing page, \$3k–\$5k B2B audit offer |
| **Week 6** *(Dec 7–Dec 13)* | **Production Cloud Serving, Staff Architecture RFC & B2B Outbound** | ⚪ Planned | Cloud Run / AWS vLLM Docker deployment, 2-page Staff Architecture RFC, 30 personalized B2B client teardowns, Phase 1 senior job application activation |

### Parked Advanced Systems Horizon (Weeks 7–10)
- **Weeks 7–8**: Evolutionary Model Merging (MergeKit, SLERP, TIES, DARE) to breed specialist models without backpropagation.  
- **Weeks 9–10**: Hardware-aware Sub-4-Bit and KV-Cache OpenAI Triton GPU kernels.

---

## 3. Resume Bullet Bank & Portfolio Proofs

### Deliverable 1: Tokenization Internals & Byte-Level BPE Kata
- *Verified Metrics*: Hand-wrote 30-line Byte-Level BPE merge loop (`src/week1_bpe_kata.py`) with **4/4 passing unit tests**, **1.36x compression** on Kubernetes log corpus, zero `<unk>` tokens on multi-byte emojis.

### Deliverable 2: Layer 18 SVD Dissection & FinOps LoRA Adapter (`Qwen/Qwen3-4B`)
- *Verified Metrics*: Profiled Layer 18 (`4:1` GQA, base Effective Rank `1,996.5–2,411.3 / 2,560`), trained rank-8 PEFT LoRA (`16,515,072` trainable params / `0.4089%` footprint) for 15 steps in `95.8s` on Apple M4 `mps` (Loss `3.8433 -> 2.0771`, $\Delta W$ nudge `0.1526%–0.2779%`), verified `merge_and_unload()` (`10.15s` vs. `21.41s` base on unseen FinOps prompt), and published 63 MB adapter + Model Card to [`imhlg/qwen3-4b-finops-lora-week1`](https://huggingface.co/imhlg/qwen3-4b-finops-lora-week1).

### Deliverable 3: 4-Bit NF4 QLoRA & `llm-compressor` GPTQ W4A16 Serving Pipeline (`imhlg/qwen3-4b-finops-w4a16-vllm`)
- *Verified Metrics (NVIDIA L4 `SM 8.9`, `bfloat16`)*: Loaded 4-bit `bitsandbytes` NF4 base in **`2.494 GB` VRAM**, trained rank-8 QLoRA for 30 steps in **`88.24s`** (`4.113 GB` peak VRAM, Loss `3.6569 -> 0.9182`), merged to `7.492 GB` BF16, and quantized via `llm-compressor GPTQ W4A16` down to **`2.47 GB` (`-67.04%` disk reduction)** (`11.95 tok/s` unpacked HF CUDA eval).

### Deliverable 4: Prototype 1 — Sub-Millisecond Residual Stream Hallucination Probe & Runtime Activation Steering (`imhlg/qwen3-4b-activation-probes-week2`)
- *Verified Metrics (NVIDIA L4, `n=1,000` FinOps Claims)*: Swept PyTorch forward hooks across Layers `[6, 12, 18, 24, 30, 35]` ([`datasets/imhlg/finops-residual-activations-1k`](https://huggingface.co/datasets/imhlg/finops-residual-activations-1k)); achieved **`1.0000` 5-fold CV & held-out (`n=200`) Precision/Recall/F1/ROC-AUC**; packaged Layer 18 linear probe + CAA steering vector into **`20.21 KB` `safetensors`**; paired with `Outlines` constrained JSON decoding at **`0.924 ms` guardrail overhead** (`49.324 ms` base vs. `50.249 ms` hooked+probed, beating `<2.0 ms` SLA and ~600x faster than external LLM judges).

---

## 4. Grok Bot Technical Interview & System Design Drills

**Grok — Drill me on these technical scenarios during mock sessions:**

1. **[Mechanistic Interpretability & Steering] The In-Flight Hallucination Interception Drill**: *"How do you attach a PyTorch forward hook to a transformer decoder layer's residual stream during generation? If an activation probe detects that a JSON extraction field is deviating from ground truth, how does adding an orthogonal steering vector alter the next token's logit distribution without crashing the KV cache in vLLM?"*  
2. **[MoE Architecture & FinOps] Structured Expert Pruning without Collapse**: *"When profiling an MoE model like OLMoE-1B-7B across an enterprise workload, why is router frequency alone insufficient to determine which experts to prune? How do you use Mean Squared Activation Norm (MSAN) alongside gating logits to safely excise weight tensors, and what loss formulation in TRL prevents catastrophic forgetting during the post-pruning distillation pass?"*  
3. **[Machine Unlearning & Compliance] Negative Preference Optimization vs. ROME**: *"Under GDPR and DPDP compliance mandates, an enterprise must erase 500 PII records memorized by a customer support LLM. Compare Negative Preference Optimization (NPO) in TRL against Rank-One Model Editing (ROME) on MLP projection layers. Which approach minimizes distortion on unrelated task capabilities?"*  
4. **[High-Throughput Systems] KV-Cache Scaling & PagedAttention Math**: *"On an AWS EC2 g5.xlarge instance (24GB VRAM), calculate the exact maximum concurrent requests a 7B FP16 model can sustain at 2,048 tokens context window under continuous batching. How does moving to PagedAttention with 4-bit KV caching alter the maximum concurrency and Time-to-First-Token (TTFT) SLA?"*

---

## 5. Daily Progress Log (Synced to Google Drive)

- **2026-11-21 (Colab Pro L4 CUDA Bursts 1 & 2 — Week 1 W4A16 Closure & Week 2 Prototype 1 Shipped)**:  
  - **Burst 1 (`imhlg/qwen3-4b-finops-w4a16-vllm`)**: NVIDIA L4 (`SM 8.9`), 4-bit NF4 load `2.494 GB`, QLoRA peak `4.113 GB` (`88.24s` / 30 steps, Loss `3.6569 -> 0.9182`), merged `7.492 GB` -> `llm-compressor GPTQ W4A16` **`2.47 GB` (`-67.04%`)**, `11.95 tok/s` CUDA eval.  
  - **Burst 2 (`imhlg/qwen3-4b-activation-probes-week2` & `datasets/imhlg/finops-residual-activations-1k`)**: 1,000-prompt Layers `[6, 12, 18, 24, 30, 35]` sweep; `1.0000` CV & held-out (`n=200`) Precision/Recall/F1/AUC; `20.21 KB` Layer 18 probe + CAA vector; `Outlines` JSON decoding + **`0.924 ms` guardrail overhead** (`49.324 ms -> 50.249 ms`).  
  - Scaffolded `src/week2_residual_hooks_kata.py` and generated Colab Burst 3 & 4 notebooks.
- **2026-11-14 (Week 1 Wrap-Up — Merged Model Verification & HF Hub Release)**:  
  - Verified unified model inference via `merge_and_unload()` (`4,022,468,096` params; `10.15s` complete `[FINOPS_AUDIT_REPORT]` vs. `21.41s` truncated base essay on unseen \$40k/mo API bill prompt).  
  - Published the 63 MB `adapter_model.safetensors`, tokenizer files, and Layer 18 SVD Model Card to [`https://huggingface.co/imhlg/qwen3-4b-finops-lora-week1`](https://huggingface.co/imhlg/qwen3-4b-finops-lora-week1) (commit `f5afdf3a`).  
- **2026-11-12 (Week 1, Days 3–5 — Local TRL LoRA SFT & $\Delta W$ Trajectory)**:  
  - Completed deep walkthrough of Layer 18 (`self_attn` + `mlp`) and LoRA rank-8 strips (`B @ A`).  
  - Executed `src/week1_svd_lora_lab.py` on M4 `mps`: trained `16,515,072` params (`0.4089%`) across 15 steps in `95.8s` (Loss `3.8433 -> 2.0771`), verified Layer 18 $\Delta W$ rank `8` (`0.1526%–0.2779%` nudge), and saved the 63 MB adapter checkpoint locally.  
- **2026-11-10 (Week 1, Days 1–2 — Layer 18 Weight Dissection & SVD)**:  
  - Added `--svd-only` mode to `src/week1_svd_lora_lab.py` and profiled Layer 18 of `Qwen/Qwen3-4B` (Effective Rank `1,996.5/2,560` on `q_proj`, `2,373.1–2,411.3/2,560` on MLP).  
- **2026-11-07 (Week 1 — MPS Inference Baseline)**:  
  - Executed `src/week1_mps_starter.py` on M4 `mps` (`bfloat16`): `4.63s` load (`7.50 GB` Metal driver memory), `129` tokens in `12.30s` (`10.5 tokens/sec`, KV-Cache length `201`, `0.03 GB` MPS tensor mem).  
- **2026-11-05 (Week 1 — Major Upgrade to v4.0)**:  
  - Upgraded learning plan and career sync to **v4.0 (`Applied AI Foundry & Weight-Level Systems Sprint`)**.  
  - Pacing anchored at **42 hours/week** (7 hrs/day Mon–Sat; Sunday rest, synthesis, and 90-min VAE/GAN review).  
  - Integrated 3-Prototype Tournament (Activation Steering, MoE Pruning, Machine Unlearning) into Weeks 2–4 with Sunday Week 4 decision gate.  
  - Parked Model Merging and Triton Kernels for Weeks 7–10.  
  - Synced to Google Drive folder `AI-FDE-Sprint-2026`.  
- **2026-11-01 (Week 1, Day 2)**:  
  - Completed hand-written Byte-Level BPE Kata (`src/week1_bpe_kata.py`) without AI autocomplete (4/4 tests passing, 1.36x compression, zero `<unk>` tokens).  
- **2026-10-31 (Week 1, Day 1)**:  
  - Local M4 environment locked (`torch 2.14.1` MPS, `transformers 5.20.2`). Linked Hugging Face Hub (`imhlg`). Executed `src/week1_tokenizer_deep_dive.py`.
