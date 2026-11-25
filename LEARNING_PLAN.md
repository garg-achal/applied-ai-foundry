# The `Applied AI Foundry` Sprint (v4.0): Weight-Level LLM Systems, Post-Training Alignment, MoE Surgery & B2B AI FinOps Foundry

**Sprint Version**: 4.0 (Strategic Evolution: Corporate Transition to AI Founder & Weight-Level Systems Mastery)  
**Target Completion**:
- **Core 6-Week Engineering & Venture Sprint (252 Hours @ 42 hrs/week)**: **Nov 1 – Dec 13, 2026** *(Phase 1 B2B Pilots & Senior Job Applications launch Nov 30 after Week 4)*  
- **Weeks 7–10 Advanced Systems Horizon (Parked)**: **Dec 14, 2026 – Jan 10, 2027** *(Evolutionary Model Merging & Custom Triton Sub-4-Bit / KV-Cache Kernels)*

**Candidate & Founder Positioning (10–13 Year Veteran — Zero Down-Leveling)**:  
10+ year Software, Cloud Infrastructure, and Technical Program leader transitioning into an **Independent AI Founder (`curons.org`) and Enterprise AI Platform Architect**, specializing in **Weight-Level LLM Optimization, Post-Training Alignment (`TRL` / `QLoRA` / `DPO` / `NPO`), Task-Specific MoE Pruning, Runtime Activation Steering, and High-Throughput `vLLM` Serving**.

**Target Strategic Tracks**:
1. **Founder & Commercial Wedge (`curons.org`)**: B2B LLM FinOps & Weight Optimization Advisory (Fixed-price 2-week optimization audit cutting enterprise GPU spend by 60–85%).  
2. **Primary Career Track A (Systems & Infra IC/Lead)**: **Senior / Staff Inference, MLOps & AI Platform Engineer**  
3. **Primary Career Track B (Staff Program & Architecture Leadership)**: **Staff / Principal Technical Program Manager — GenAI / AI Platform**  
4. **Secondary Career Track C (Model Customization)**: **Lead / Senior Applied AI & Post-Training Engineer (Fine-Tuning, Evals & Agents)**  
5. **Secondary Career Track D (Customer Architecture)**: **Forward Deployed Engineer (FDE) / Principal AI Solutions Architect**

---

## Executive Summary: The 4 Core Pillars & 3-Prototype Startup Tournament (v4.0)

Instead of building fragile, low-moat API wrappers, v4.0 operates directly at the **weight, neuron, and activation level** across 4 core engineering pillars, culminating in a **3-Prototype Tournament** to launch a high-defensibility venture:

```text
[Raw Enterprise Data] ──> [Synthetic Data / Distilabel] ──> [Hugging Face Dataset]
                                                                   │
                                                                   ▼
[Merged Weights / Pruned Checkpoint] <── [Weight Surgery / LoRA] <── [TRL SFTTrainer + PEFT]
          │
          ▼
[vLLM Inference Server] (PagedAttention + Forward Hook Activation Steering)
          │
          ▼
[Async Load Benchmark Engine] ──> [FinOps Unit-Cost & Latency ROI vs Frontier APIs]
```

### The 3-Prototype Tournament (Weeks 2–4)
1. **Week 2 / Prototype 1 (Idea 1): Runtime Activation Steering & Production Probe Classifiers**  
   - *Core Mechanics*: Attach PyTorch forward hooks (`register_forward_hook`) and Sparse Autoencoder (SAE) probes to hidden-layer residual streams in `vLLM` to catch hallucinations and policy breaches in real time before tokens are emitted.  
   - *Moat*: Runtime activation clamping with zero prompt-latency overhead; high enterprise defensibility against wrapper commoditization.  
2. **Week 3 / Prototype 2 (Idea 2): Task-Specific MoE Expert Surgery & Pruning**  
   - *Core Mechanics*: Profile router gate logits on an open Mixture-of-Experts checkpoint (`OLMoE-1B-7B` / `Qwen2.5-MoE`), excise underutilized experts (<2% activation) via Mean Squared Activation Norm (MSAN), and heal the pruned graph with a 1-epoch TRL distillation recovery pass.  
   - *Moat*: High FinOps ROI—runs large MoE models on single consumer/edge GPUs (cutting server footprint by 40–60%).  
3. **Week 4 / Prototype 3 (Idea 3): Surgical Machine Unlearning & Weight Editing**  
   - *Core Mechanics*: Subspace weight projection, rank-one model editing (ROME/MEMIT), and Negative Preference Optimization (NPO) in `TRL` to surgically delete memorized sensitive customer records/PII without full retraining, maintaining >99% task accuracy.  
   - *Moat*: High regulatory lock-in for GDPR/DPDP/EU AI Act compliance with cryptographic erasure audit certificates.  
4. **End of Week 4 Tournament Lock**:  
   - The Sunday 5:30 PM diagnostic selects the single winning prototype based on empirical benchmark results and enterprise customer demand to package into a paid commercial offering.

### Parked Post-6-Week Horizon (Weeks 7–10)
- **Idea 4 (Weeks 7–8): Evolutionary Model Merging & Weight-Space Foundry** (Task-vector arithmetic, SLERP, TIES, DARE via `MergeKit` without backpropagation).  
- **Idea 5 (Weeks 9–10): Hardware-Aware Sub-4-Bit & KV-Cache Triton Kernels** (Overcoming the GPU memory bandwidth wall with custom OpenAI Triton GEMM kernels).

---

## 1. Operating Rules & Time Commitment (Strict 42 Hours / Week)

To protect deep cognitive processing time and internalize the underlying tensor linear algebra, weekly effort is capped at **42 hours/week** across a 6-day sprint (Monday through Saturday, 7 hours/day), reserving Sunday for rest, synthesis, and interview review.

### Daily 7-Hour Schedule Breakdown
- **Block 1 — Theory & Tensor Dissection (2.5 hrs/day)**: Step through transformer layers in PyTorch notebooks; inspect weight matrices ($W_q, W_k, W_v, W_o$, MLP), gradients, and Singular Value Decomposition (SVD) trajectories before touching training scripts.  
- **Block 2 — Focused Hands-On Lab Build (3.5 hrs/day)**: Implement and benchmark one self-contained production module per day (`TRL`, PyTorch hooks, `vLLM`, pruning scripts) without multitasking.  
- **Block 3 — Synthesis, Telemetry & GTM (1.0 hr/day)**: Log architectural takeaways in notes, track daily GPU spend, and calibrate for bi-weekly Thursday/Sunday diagnostics.  
- **Sunday Recovery & Theory Bridge (3.0 hrs)**: Weekly synthesis, diagnostic calibration, and a dedicated **90-minute Sunday review block** covering classic generative architectures (Variational Autoencoders / VAEs, ELBO math, reparameterization trick, GAN minimax games) to bridge recruiter screening checklists while keeping weekdays 100% focused on autoregressive LLMs.

---

## 2. Full-Fledged Hardware & Tooling Stack

| Layer | Tool / Platform | Role in the v4.0 Sprint & Founder Discipline |
| :--- | :--- | :--- |
| **IDE & Agent Orchestration** | **VS Code IDE + Antigravity Desktop App (Right-Pane Terminal) + Claude Code CLI (`claude`)** | Code editing in **VS Code**, agent orchestration & right-pane terminal execution in **Antigravity Desktop App**, terminal agent workflows in **Claude Code CLI** (`CLAUDE.md`, `claude mcp add`, `PreToolUse` hooks). Hand-write core algorithms; pass Explain-Back Gates before merging. |
| **Research & Theory Brain** | **NotebookLM Plus (`AI + LLM + Deep Learning Plan`) + Gemini Guided Learning** | Notebooks loaded with canonical papers/docs. Multi-modal synthesis, spaced retrieval flashcards, and scenario drills. |
| **Local Tier-0 GPU (M4 Mac)** | **MacBook Air M4 (24GB Unified Memory, ~16–18GB usable GPU limit)** | Daily zero-cost workhorse using **PyTorch `mps`**, **Apple `mlx-lm`**, and **`llama.cpp` (Metal)**. Primary lab models: **`Qwen/Qwen3-4B`**, **`Qwen/Qwen2.5-7B-Instruct`**, **`OLMoE-1B-7B`**, and **`Llama-3.1-8B-Instruct`**. |
| **Local High-Speed Agent & Teacher LLM** | **LM Studio (`Qwen3-Coder-30B-A3B` `Q3_K_L` GGUF, ~14.5GB)** | Local OpenAI-compatible server (`http://localhost:1234/v1`) for MCP, tool calling, and Teacher $\rightarrow$ Student synthetic data generation. Unload (`lms unload --all`) before running heavy PyTorch `mps` jobs. |
| **Cloud Tier-1 GPU Compute** | **Google Colab Pro (200 CU/mo) + AWS EC2 (`g5.xlarge` A10G 24GB)** | Bursts on **NVIDIA L4 / A10G** for `bitsandbytes` NF4 QLoRA, `llm-compressor` (W4A16/FP8), and **`vLLM`** high-concurrency multi-request load benchmarks. |
| **API & Cloud Hosting Plane** | **Google AI Studio + HF Spaces + GCP Cloud Run** | Zero-cost hosting for demo endpoints on **Hugging Face Spaces** and **GCP Cloud Run**. Banked Google AI Pro credits reserved for burst evaluations. |
| **Venture & Portfolio Asset** | **`curons.org` + GitHub (`garg-achal/applied-ai-foundry`) + Hugging Face (`garg-achal`)** | Public proof-of-work repository, verifiable benchmark dashboards, and commercial B2B audit landing page. |
| **Career & Interview Sync** | **Google Drive (`gdrive`) + Grok Bot** | Daily sync in Google Drive folder `AI-FDE-Sprint-2026` (`1_IHEWhtTKvCpsOoPqGL36_G_42Gc8qXO`): `LEARNING_PLAN.md` (`1DTY2snGS1CdS-VAa23luE4wC5p4X7JwMGXMpahVESeQ`) and `GROK_CAREER_AND_PROGRESS_SYNC.md` (`1YwnHeVfQye214j5P_o3ID_NiGCQAJRLN1MTzKQhfy1g`). |

---

## 3. 6-Week Deep Mastery Curriculum & Calendar Timeline (252 Hours Total)

### Week 1 (Nov 1 – Nov 8 | 42 hrs): Tokenization Internals, Transformer Weight Mechanics ($W_q, W_k, W_v, W_o$, MLP) & SVD
**Milestone Focus**: Move past treating models as black boxes; inspect internal parameter matrices and verify low-rank updates.
- [x] Local PyTorch `mps` starter pipeline (`src/week1_mps_starter.py`) with Metal memory telemetry & `torch.mps.synchronize()`.  
- [x] Cloud Google Colab `cuda` starter pipeline with dynamic dtype selection (`bf16`/`fp16`) & VRAM profiling.  
- [x] **Scratch Byte-Level BPE Kata (`src/week1_bpe_kata.py`)**: 4/4 passing unit tests, 1.36x compression on K8s telemetry corpus, zero `<unk>` tokens on multi-byte UTF-8 emojis.  
- [x] **Chat Templates & Left-Padding**: Verifying `apply_chat_template` with thinking budgets and proving why decoder-only batching strictly requires `padding_side="left"` + attention masks.  
- [x] **Day 1–2 Theory & Lab (Weight Dissection & SVD)**: Dissected Layer 18 attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`, 4:1 GQA) and MLP (`gate_proj`, `up_proj`, `down_proj`) tensors in PyTorch; ran SVD to verify high base Effective Rank (`1,996.5–2,411.3 / 2,560`).  
- [x] **Day 3–4 Lab (Local TRL LoRA & Colab CUDA 4-Bit NF4 QLoRA)**: Ran `src/week1_svd_lora_lab.py` on M4 `mps` (`bfloat16`) via `trl.SFTTrainer` + PEFT LoRA (`r=8, alpha=16`); trained `16,515,072` params (`0.4089%`) for 15 steps in `95.8s` (Loss `3.8433 -> 2.0771`). Executed **Colab Burst 1** on **NVIDIA L4 (`SM 8.9`, `bfloat16`)**: 4-bit `bitsandbytes` NF4 load `2.494 GB`, QLoRA peak `4.113 GB` VRAM, 30 steps in `88.24s` (Loss `3.6569 -> 0.9182`), merged `7.492 GB` -> **`llm-compressor` GPTQ W4A16 `2.47 GB` (`-67.04%`)** (`11.95 tok/s` CUDA eval), published to [`garg-achal/qwen3-4b-finops-w4a16-vllm`](https://huggingface.co/garg-achal/qwen3-4b-finops-w4a16-vllm).  
- [x] **Day 5 Lab (Adapter Tensor Trajectories & Merge Verification)**: Verified $\Delta W = \frac{\alpha}{r} B \cdot A$ (`0.1526%–0.2779%` nudge, rank `8`), tested `merge_and_unload()` (`4,022,468,096` params, `10.15s` vs. `21.41s` base on unseen FinOps prompt), and published 63 MB adapter + SVD Model Card to [`garg-achal/qwen3-4b-finops-lora-week1`](https://huggingface.co/garg-achal/qwen3-4b-finops-lora-week1) (commit `f5afdf3a`).  
- [x] **Week 1 Deliverable (100% Complete — Local M4 + Colab CUDA W4A16 + HF Hub)**: Standalone SVD & LoRA script (`src/week1_svd_lora_lab.py`), Colab NF4 QLoRA + `llm-compressor` GPTQ W4A16 checkpoint ([`garg-achal/qwen3-4b-finops-w4a16-vllm`](https://huggingface.co/garg-achal/qwen3-4b-finops-w4a16-vllm)), and BF16 adapter ([`garg-achal/qwen3-4b-finops-lora-week1`](https://huggingface.co/garg-achal/qwen3-4b-finops-lora-week1)). *(Pending: Bi-weekly diagnostic Q2–Q8).*

### Week 2 (Nov 9 – Nov 15 | 42 hrs): Residual Streams, Runtime Activation Steering & Linear Probe Classifiers (Startup Idea 1)
**Milestone Focus**: Build Prototype 1—intercept internal hidden states during live inference to catch hallucinations before token emission.
- [x] **PyTorch Forward Hooks**: Scaffolded Write-First Lab 1 kata (`src/week2_residual_hooks_kata.py` — local hand-written gate pending) and ran batched `register_forward_hook` sweep across Layers `[6, 12, 18, 24, 30, 35]` on Colab L4 (`notebooks/colab_burst2_activation_probe_1k.ipynb`).  
- [x] **Guided Structured Generation**: Integrated `Outlines` (`outlines_mode="outlines"`) grammar-constrained JSON decoding (`FinOpsAuditSchema`) at the logit level alongside Layer 18 runtime steering.  
- [x] **Activation Dataset Generation**: Synthesized 1,000 contrastive FinOps claims (500 grounded vs. 500 hallucinated; 800 train / 200 held-out test), captured `[1000, 2560]` residual activations on NVIDIA L4, and published [`datasets/garg-achal/finops-residual-activations-1k`](https://huggingface.co/datasets/garg-achal/finops-residual-activations-1k).  
- [x] **Linear Probe Classifier**: Trained Layer `[6, 12, 18, 24, 30, 35]` `LogisticRegression` probes (`1.0000` 5-fold CV & held-out `n=200` test Precision / Recall / F1 / ROC-AUC) and saved Layer 18 probe + CAA vector (`20.21 KB` `safetensors`).  
- [x] **Runtime Activation Clamping**: Implemented Contrastive Activation Addition ($\vec{v}_{\text{truth}}$) clamping `h_18 <- h_18 + alpha * v_truth` in-flight when $P(\text{halluc} \mid \mathbf{h}_{18}) > 0.5$.  
- [x] **Week 2 Deliverable (Prototype 1 — Colab L4 Shipped)**: Published [`garg-achal/qwen3-4b-activation-probes-week2`](https://huggingface.co/garg-achal/qwen3-4b-activation-probes-week2) with **`0.924 ms` guardrail overhead** (`49.324 ms` base -> `50.249 ms` hooked+probed on L4, beating `<2.0 ms` SLA and ~600x faster than external LLM judges). *(Complete local 25-line kata + Explain-Back Gate in `src/week2_residual_hooks_kata.py`).*

### Week 3 (Nov 16 – Nov 22 | 42 hrs): MoE Router Profiling, Structured Expert Pruning & Distillation (Startup Idea 2)
**Milestone Focus**: Build Prototype 2—profile Mixture-of-Experts routing gates, prune inactive experts, and recover accuracy.
- [ ] **MoE Architecture Deep Dive**: Inspect top-$k$ router softmax gating mechanisms in an open MoE model (`OLMoE-1B-7B` or `Qwen2.5-MoE`).  
- [ ] **Router Gate Profiling**: Pass 2,000 domain-specific examples through the model; compute the Mean Squared Activation Norm (MSAN) and routing frequency across all experts.  
- [ ] **Surgical Expert Pruning**: Identify experts utilized <2% of the time; physically excise their weight tensors from the checkpoint and adjust router dimension heads.  
- [ ] **TRL Knowledge Distillation Recovery**: Run a 1-epoch recovery pass using `TRL` SFT on the pruned checkpoint to restore boundary reasoning capabilities.  
- [ ] **Throughput & VRAM Profiling**: Benchmark the slimmed checkpoint against the dense/unpruned baseline in `vLLM`—measuring active VRAM reduction and tokens/sec gains.  
- [ ] **Week 3 Deliverable (Prototype 2)**: Automated MoE Pruning & Distillation pipeline with verified VRAM reduction (-35% VRAM footprint, <1.5% accuracy drop).

### Week 4 (Nov 23 – Nov 29 | 42 hrs): Machine Unlearning, High-Throughput `vLLM` & Thesis Tournament Lock (Startup Idea 3)
**Milestone Focus**: Build Prototype 3—surgically erase sensitive data via NPO; benchmark in `vLLM`; lock commercial venture thesis.
- [ ] **Synthetic Memorization Injection**: Inject 100 identifiable synthetic enterprise records into a 7B model until memorized (100% extraction recall).  
- [ ] **Negative Preference Optimization (NPO)**: Implement NPO and gradient ascent loss in `TRL` targeting only the forget set, penalizing log-likelihood of memorized records while anchoring on retain data.  
- [ ] **Subspace Weight Editing**: Test rank-one model editing (ROME) on MLP projection layers as an alternative factual erasure mechanism.  
- [ ] **High-Throughput Serving with `vLLM`**: Deploy the model on `vLLM` with PagedAttention, continuous batching, and chunked prefill; run async concurrency load tests (`httpx`/Locust).  
- [ ] **Sunday 5:30 PM Tournament Decision Gate**: Compare all 3 prototypes (Activation Steering vs. MoE Pruning vs. Machine Unlearning) across technical robustness, FinOps savings, and customer validation. Lock the single startup thesis for Weeks 5–6.  
- [ ] **Week 4 Deliverable (Prototype 3 & Thesis Lock)**: Unlearning benchmark (0% recall on forget set, >99% retained JSON accuracy) + 1-page Thesis Selection Memo.

### Week 5 (Nov 30 – Dec 6 | 42 hrs): Productizing the Winning Thesis, Open-Source Wedge & Commercial Entity Setup
**Milestone Focus**: Turn the winning prototype into an enterprise-ready open-source asset and launch B2B advisory offering.
- [ ] **Packaging the Engine**: Refactor the winning prototype into a clean, drop-in Python CLI or `vLLM` plugin repository (`applied-ai-foundry`).  
- [ ] **Automated CI/CD & Evaluation Gates**: Configure GitHub Actions to run deterministic validation tests on every commit using synthetic golden datasets.  
- [ ] **FinOps Calculator & Unit Economics Suite**: Build a comprehensive TCO calculator comparing blended cost per million tokens and P95 latency against Claude 3.5 Sonnet and GPT-4o.  
- [ ] **Commercial Entity & Branding**: Configure `curons.org` landing page showcasing interactive benchmark telemetry and the "2-Week LLM FinOps & Weight Optimization Audit" pilot (\$3,000–\$5,000 fixed price).  
- [ ] **Week 5 Deliverable**: Public GitHub repository with reproducible benchmarks + Hugging Face model card + Live `curons.org` pilot offer.

### Week 6 (Dec 7 – Dec 13 | 42 hrs): Enterprise Production Deployment, Staff Architecture RFC & B2B Pilot Outbound
**Milestone Focus**: Deploy production endpoints, publish the flagship 2-Page Architecture RFC, and close the first design partner pilot.
- [ ] **Production Cloud Deployment**: Package the optimized model into a multi-stage Docker container; deploy to Hugging Face Spaces (demo) and AWS EC2 / GCP Cloud Run with autoscaling rules.  
- [ ] **Staff AI Platform Architecture RFC**: Author a 2-page enterprise architecture proposal detailing GPU sizing, KV-cache memory limits, and the weight-level optimization pipeline.  
- [ ] **Targeted B2B Outbound Campaign**: Identify 30–50 engineering leaders (VPs of Engineering, Heads of AI Infra) running high-volume LLM workloads; send personalized unit-cost and latency teardowns.  
- [ ] **Phase 1 Senior Career Pipeline Launch**: Activate job alert pipelines for Staff AI Platform TPM, Senior Inference Systems Engineer, and Principal FDE roles using customized resume variants.  
- [ ] **Week 6 Deliverable**: Deployed live platform + 2-Page Staff RFC + 30 custom outbound pitches + first paid B2B pilot pipeline.

---

## 4. Post-6-Week Advanced Systems Horizon (Weeks 7–10 — Parked)

### Weeks 7–8: Evolutionary Model Merging & Weight-Space Foundry (Idea 4)
- Task-vector arithmetic ($\tau = \theta_{\text{ft}} - \theta_{\text{base}}$), TIES-Merging (trimming, sign resolution), DARE (Drop and Rescale), and SLERP on non-Euclidean parameter spheres.  
- Automated evolutionary search loops using `MergeKit` to breed hybrid specialist models without backpropagation.

### Weeks 9–10: Hardware-Aware Sub-4-Bit & KV-Cache Triton Kernels (Idea 5)
- Breaking the GPU memory bandwidth wall ($\text{FLOPS}/\text{byte}$ arithmetic intensity).  
- Writing block-level parallel loops and shared-memory loads in OpenAI Triton for custom compressed KV-cache retrieval and non-uniform sub-4-bit GEMM.

---

## 5. Bi-Weekly Assessment & Telemetry Cadence

Assessments occur twice a week on **Thursdays and Sundays at 5:30 PM IST** using the 8-question hybrid framework:
- **Part 1 (Diagnostics Q1–Q4)**: Hardware utilization, agent tooling configurations, token spend/routing, and shipped code artifacts.  
- **Part 2 (Technical Scenarios Q5–Q8)**: Apple Silicon KV-cache sizing, schema-constrained decoding, prefix cache hit-rate mechanics, and AWS GPU batch economics.  
- **Current State (Thursday, Nov 5, 2026)**: Q1 logged (**A**); Q2 pending response.

---

## 6. Daily Progress Log

- **2026-11-21 (Colab Pro L4 CUDA Bursts 1 & 2 — Week 1 W4A16 Closure & Week 2 Prototype 1 Activation Probe Engine Shipped)**:  
  - **Burst 1 (`garg-achal/qwen3-4b-finops-w4a16-vllm`)**: Executed 4-bit `bitsandbytes` NF4 + Double Quantization QLoRA (`r=8, alpha=16`) on NVIDIA L4 (`SM 8.9`, `bfloat16`): `2.494 GB` NF4 load, `4.113 GB` peak training VRAM, 30 steps in `88.24s` (Loss `3.6569 -> 0.9182`). Merged to `7.492 GB` BF16 and quantized via `llm-compressor GPTQ W4A16` to **`2.47 GB` (`-67.04%` disk reduction)**; verified CUDA eval (`11.95 tok/s`, `9.144 GB` unpacked `transformers` peak VRAM).  
  - **Burst 2 (`garg-achal/qwen3-4b-activation-probes-week2` & `datasets/garg-achal/finops-residual-activations-1k`)**: Ran batched `register_forward_hook` residual sweep across Layers `[6, 12, 18, 24, 30, 35]` on 1,000 FinOps claims (`800` train / `200` held-out test); achieved **`1.0000` Precision/Recall/F1/ROC-AUC**, packaged Layer 18 probe + CAA steering vector (`20.21 KB` `safetensors`), integrated `Outlines` JSON constrained decoding (`outlines_mode="outlines"`), and benchmarked **`0.924 ms` guardrail overhead** (`49.324 ms` base -> `50.249 ms` hooked+probed, beating `<2.0 ms` SLA).  
  - Scaffolded local Write-First Lab 1 kata (`src/week2_residual_hooks_kata.py`) and generated Colab Burst 3 & 4 notebooks.
- **2026-11-14 (Week 1 Wrap-Up — Merged Model Verification & HF Hub Release)**:  
  - Verified unified model inference after `merge_and_unload()` on an unseen \$40k/mo API bill prompt (`4,022,468,096` params; merged model returned a complete 3-line `[FINOPS_AUDIT_REPORT]` in `10.15s` vs. `21.41s` truncated base output).  
  - Published the 63 MB `adapter_model.safetensors`, tokenizer files, and Layer 18 SVD Model Card to [`https://huggingface.co/garg-achal/qwen3-4b-finops-lora-week1`](https://huggingface.co/garg-achal/qwen3-4b-finops-lora-week1) (commit `f5afdf3a`).  
- **2026-11-12 (Week 1, Days 3–5 — Local TRL LoRA SFT & $\Delta W$ Trajectory)**:  
  - Completed deep top-down walkthrough of Transformer layers (`self_attn` + `mlp` rooms) and LoRA rank-8 adapters (`B @ A`).  
  - Executed `src/week1_svd_lora_lab.py` on M4 `mps` (`bfloat16`): trained `16,515,072` params (`0.4089%`) across 15 steps in `95.8s` (Loss `3.8433 -> 2.0771`), verified Layer 18 $\Delta W$ rank `8` (`0.1526%–0.2779%` relative nudge), and saved the 63 MB adapter checkpoint locally.  
- **2026-11-10 (Week 1, Days 1–2 — Layer 18 Weight Dissection & SVD)**:  
  - Added `--svd-only` flag to `src/week1_svd_lora_lab.py` and profiled Layer 18 of `Qwen/Qwen3-4B` (4:1 GQA ratio; Effective Rank `1,996.5/2,560` on `q_proj` and `2,373.1–2,411.3/2,560` on MLP projections).  
- **2026-11-07 (Week 1 — MPS Inference Baseline)**:  
  - Executed `src/week1_mps_starter.py` on M4 `mps` (`bfloat16`): `4.63s` load (`7.50 GB` Metal driver memory), generated `129` tokens in `12.30s` (`10.5 tokens/sec`, KV-Cache length `201`, `0.03 GB` MPS tensor mem).  
- **2026-11-05 (Week 1 — Major Upgrade to v4.0 `Applied AI Foundry` & Venture Roadmap)**:  
  - Upgraded sprint to **v4.0** following career transition to independent AI founder (`curons.org`) and senior systems engineer.  
  - Locked strict **42-hour weekly budget** (7 hrs/day Mon–Sat; Sunday rest, synthesis, and 90-min VAE/GAN review).  
  - Integrated the **3-Prototype Tournament** (Activation Steering, MoE Pruning, Machine Unlearning) into Weeks 2–4 with a Sunday Week 4 decision gate.  
  - Parked Evolutionary Model Merging (Weeks 7–8) and Triton Sub-4-Bit Kernels (Weeks 9–10) as post-sprint milestones.  
  - Preserved existing Week 1 lab milestones (Byte-Level BPE kata completed with 4/4 tests passed and 1.36x compression).  
- **2026-11-01 (Week 1, Day 2)**:  
  - Completed hand-written Byte-Level BPE Kata (`src/week1_bpe_kata.py`) without AI autocomplete (4/4 tests passing, 1.36x compression on K8s telemetry corpus, zero `<unk>` on multi-byte emojis). Initialized `applied-ai-foundry` repository.  
- **2026-10-31 (Week 1, Day 1)**:  
  - Verified local `.venv` on M4 MacBook Air (`torch 2.14.1` MPS, `transformers 5.20.2`, `accelerate 1.15.0`, `tokenizers 0.23.3`). Linked Hugging Face Hub profile (`garg-achal`). Executed `src/week1_tokenizer_deep_dive.py` on `Qwen/Qwen3-4B`.
