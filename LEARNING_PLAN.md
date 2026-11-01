# 6-Week Intensive AI Engineering, MLOps & FDE Sprint Plan (v3)

**Target Completion**: Mid-December 2026 (6 Weeks Core Mastery @ 25 hrs/week = 150 Hours; Optional 2-Week FDE Interview Buffer through End of Dec 2026)  
**Learner Profile**: Tech professional with 10 years of experience in software & infrastructure, foundational Python skills, transitioning into advanced AI Systems Engineering, MLOps, and **Forward Deployed Engineer (FDE) / Enterprise AI Solutions Architect (Pre-Sales)** roles.  
**Mentor Persona**: Expert Senior MLOps, AI Systems & Forward Deployed Engineer (production-grade, modular, hardware-aware code; zero "vibe-coding" atrophy via Write-First & Explain-Back gates; FDE business/TCO framing).

---

## Executive Summary: The 4 Core Pillars of What We Are Learning

Instead of treating LLMs like a black-box API, this sprint opens the hood on the complete open-weight & enterprise AI engineering stack across 4 core pillars (accelerated into **6 weeks @ 25 hrs/week = 150 hours**):

1. **Pillar 1 — Advanced Inference, Tokenization & Agentic Tooling (Weeks 1–2 / 50 hrs)**  
   How LLMs process text and generate tokens under the hood using Hugging Face `transformers` v5 (`Qwen/Qwen3-4B`, `Gemma 3/4`) on your **M4 Mac (`mps`)** and **Google Colab (`cuda`)**:
   - **Tokenization Internals**: How raw text becomes subword IDs (Byte-Level BPE), how `apply_chat_template` formats system/user/thinking/tool roles into control tokens, and how **left-padding + attention masks** let GPUs batch multiple prompts safely.
   - **Hardware-Accelerated Inference & KV-Cache**: Loading models in 16-bit precision (`bfloat16`), running forward passes with `torch.inference_mode()`, inspecting `DynamicCache`/`StaticCache`, custom `LogitsProcessor`, and measuring real token/sec throughput and VRAM usage.
   - **Constrained Decoding & Custom MCP Servers (FDE Core)**: Guaranteeing strict JSON/Pydantic outputs and building custom **Model Context Protocol (`FastMCP`)** servers wired to your local `Qwen3-Coder-30B-A3B` (`Q3_K_L`) in LM Studio.
2. **Pillar 2 — Eval-Driven Development & Parameter-Efficient Fine-Tuning: PEFT / LoRA / DPO / GRPO (Weeks 3–4 / 50 hrs)**  
   Building Golden Datasets + automated **LLM-as-a-Judge** evaluation harnesses (`promptfoo` / `DeepEval`) first, then curating datasets (`datasets`) and fine-tuning open-weight models locally on M4 Unified Memory (`mlx-lm` / `peft`) and on Colab GPUs (`bitsandbytes` 4-bit `NF4` QLoRA + `trl` v1+ `SFTConfig`, `DPOConfig`, and `GRPOConfig`), followed by adapter merging and publishing to Hugging Face Hub.
3. **Pillar 3 — Post-Training Quantization, High-Throughput Serving & CTO TCO Math (Week 5 / 25 hrs)**  
   Compressing models (`FP16/BF16` $\rightarrow$ `INT8`/`INT4`/`FP8` via `GGUF` and `llm-compressor`) and serving them at high speed with `llama.cpp` (local M4 Metal / LM Studio) and `vLLM` (cloud CUDA PagedAttention, continuous batching, prefix caching, speculative decoding), backed by a **CTO GPU Sizing & Token Economics (TCO) Calculator**.
4. **Pillar 4 — Production Hybrid Agentic RAG, LLMOps Observability & FastAPI Microservices (Week 6 / 25 hrs)**  
   Building hybrid retrieval (Dense Vector Embeddings + Sparse BM25 + Cross-Encoder Reranking evaluated with `RAGAS`), instrumenting with OpenTelemetry GenAI tracing (`Arize Phoenix` / `Langfuse`) and CI eval gates, and wrapping everything into an async, containerized `FastAPI` microservice deployed for \$0 on **Google Cloud Run (Always-Free CPU Tier)** and **Hugging Face Spaces**.

---

## 1. Full-Fledged Hardware & Tooling Stack

| Layer | Tool / Platform | Role in the v3 Sprint & Budget Discipline |
| :--- | :--- | :--- |
| **IDE & Agent Orchestration** | **Antigravity IDE** | Primary AI-first IDE (Sidebar Agent, `Cmd+I` inline edits, Supercomplete, `.agents` rules). **Anti-Vibe-Coding Rule**: Hand-write/pseudocode core 25-line algorithms and pass the **Explain-Back Gate** before merging. |
| **Research & Theory Brain** | **NotebookLM Plus + Gemini Guided Learning** | 1 Notebook per week loaded with canonical papers/docs. Use **Mind Maps, Audio Overviews (Critique/Debate), Flashcards, and Quizzes** (+3d / +10d spaced retrieval). |
| **Local Tier-0 GPU (M4 Mac)** | **MacBook Air M4 (24GB Unified Memory, ~16–18GB usable GPU limit)** | Daily zero-cost workhorse using **PyTorch `mps`**, **Apple `mlx-lm`**, and **`llama.cpp` (Metal)**. Primary open-weight lab models: **`Qwen/Qwen3-4B`** (Apache-2.0) and **`google/gemma-3-4b-it` / `gemma-4-E4B-it`**. |
| **Local High-Speed Agent LLM** | **LM Studio (`Qwen3-Coder-30B-A3B` `Q3_K_L` GGUF, ~14.5GB)** | Local OpenAI-compatible server (`http://localhost:1234/v1`) for MCP, tool-calling, and RAG dev. **Memory Rule**: Run `lms unload --all` before running raw PyTorch `mps` or `mlx-lm` training jobs. |
| **Cloud Tier-1 GPU Compute** | **Google Colab Pro (200 Compute Units / mo)** | Connect in short 20–30 min bursts (then disconnect immediately) on **NVIDIA T4 (~1.5 CU/hr)** and **NVIDIA L4 (~4.8 CU/hr, Ada SM 8.9)** for `bitsandbytes` NF4 QLoRA, **`llm-compressor`** (W4A16/FP8), and **`vLLM`** benchmarks. |
| **API & Cloud Hosting Plane** | **Google AI Studio + HF Spaces + GCP Cloud Run** | Use **AI Studio Free Tier** + local 30B model first so your **\$10/mo Google AI Pro GCP credits** stay banked and rolling over. Host 24/7 portfolio apps on **Hugging Face Spaces (Free 16GB RAM)** and **GCP Cloud Run (Always-Free CPU Tier)**. |
| **Career & Interview Sync** | **Google Drive (`gdrive`) + Grok Bot** | Weekly in-place sync of `LEARNING_PLAN.md` (`1wx9T1Z6gVDa3z9Ax9PrmivjHdVacT3QJ`) and `GROK_CAREER_AND_PROGRESS_SYNC.md` (`198pc1nNyggj3TJ6djKW0PagPXuIGEC0w`) in Google Drive folder `AI-FDE-Sprint-2026` (`1_IHEWhtTKvCpsOoPqGL36_G_42Gc8qXO`). |

---

## 2. 6-Week Deep Mastery Curriculum (150 Hours @ 25 hrs/wk)

### Week 1 (25 hrs): Tokenization Internals, `transformers` v5 Inference & KV-Cache Architecture *(Current)*
**Canonical References**: Karpathy *Let's Build the GPT Tokenizer* (`minbpe`), HF LLM Course Ch. 1–2 & 6, `transformers` v5 Cache docs.
- [x] Local PyTorch `mps` starter pipeline (`src/week1_mps_starter.py`) with Metal memory telemetry & `torch.mps.synchronize()`
- [x] Cloud Google Colab `cuda` starter pipeline with dynamic dtype selection (`bf16`/`fp16`) & VRAM profiling
- [x] **Deep Dive into `AutoTokenizer` & Scratch BPE Kata (`src/week1_bpe_kata.py`, `src/week1_tokenizer_deep_dive.py`)**: BPE subword splits, byte-level fallback (0..255), and hand-written 30-line BPE merge loop with 4/4 passing tests & 1.36x compression
- [x] **Chat Templates & Reasoning Control**: Inspecting `apply_chat_template` with `Qwen/Qwen3-4B` (`enable_thinking=True/False`, `tools=` injection) and why Decoder-only batching requires `padding_side="left"` + attention masks
- [ ] **Decoding & Sampling Internals**: Greedy vs. Beam Search, Temperature scaling, Top-$k$, Top-$p$ (Nucleus), Repetition penalties, and writing a custom `LogitsProcessor`
- [ ] **KV-Cache Mechanics & Streaming**: `transformers` v5 `DynamicCache` vs. `StaticCache` memory growth math, `TextIteratorStreamer`, and `model.generate_batch()`

### Week 2 (25 hrs): Constrained Decoding, Agentic Tool Calling & Custom MCP Servers (FDE Core)
**Canonical References**: Model Context Protocol (`modelcontextprotocol.io`) docs, Kaggle 5-Day AI Agents whitepapers, Chip Huyen *AI Engineering* (Agents & Structured Outputs).
- [ ] **Constrained & Structured Decoding**: Enforcing strict Pydantic / JSON schemas at the logit level (`outlines` / grammar-constrained generation) with zero syntax failures
- [ ] **Native Function Calling & ReAct Loops**: Implementing multi-step tool calling, error recovery, and explicit state machines (`LangGraph` / `Google ADK`) against local `Qwen3-Coder-30B-A3B` (LM Studio) and `Qwen3-4B` (`mps`)
- [ ] **Model Context Protocol (MCP) Engineering**: Building a custom, production-grade **MCP Server (`FastMCP`)** exposing enterprise data/tools with input validation, RBAC/least-privilege boundaries, and audit logging
- [ ] **Week 2 FDE Deliverable**: End-to-end local Agent + Custom MCP Server repo with architecture diagram & FDE Customer Objection write-up

### Week 3 (25 hrs): Eval-Driven Development & Local PEFT / LoRA Fine-Tuning on Apple Silicon
**Canonical References**: Hamel Husain & Shreya Shankar *AI Evals Field Guide*, HF `smol-course`, Apple `mlx-lm` LoRA documentation.
- [ ] **Eval-First Engineering**: Curating domain datasets with HF `datasets`, generating a 100-item **Golden Dataset**, and building a binary pass/fail **LLM-as-a-Judge & deterministic eval suite** (`promptfoo` / `DeepEval`) *before* fine-tuning
- [ ] **PEFT / LoRA Architecture & Memory Math**: Deriving exact VRAM/Unified Memory budgets for weights, LoRA adapters ($r$, $\alpha$, target projection modules), gradients, optimizer states, and activations
- [ ] **Local Fine-Tuning on M4 Unified Memory (Path A — Apple `mlx-lm`)**: Fast 4-bit QLoRA fine-tuning of `Qwen3-4B` natively on Apple Silicon Metal arrays
- [ ] **Local Fine-Tuning on M4 Unified Memory (Path B — PyTorch `mps`)**: `peft` + `trl` v1+ `SFTTrainer` using modern `SFTConfig` (`completion_only_loss=True`, sequence packing, `bfloat16`)

### Week 4 (25 hrs): Cloud QLoRA (`NF4`), Preference Tuning (`DPO` / `GRPO`) & Hub Publishing
**Canonical References**: *LoRA* & *QLoRA (NF4)* papers in NotebookLM, HF `trl` v1+ documentation (`SFTConfig`, `DPOConfig`, `GRPOConfig`).
- [ ] **Cloud QLoRA on Colab CUDA (L4 GPU)**: 4-bit NormalFloat (`nf4`), double quantization, and paged AdamW optimizers with `bitsandbytes` + `trl`
- [ ] **Preference & RL Tuning (2026 Standard)**: Aligning the SFT model using **Direct Preference Optimization (`DPOConfig`)** and **Group Relative Policy Optimization (`GRPOConfig`)** with verifiable code/format reward functions
- [ ] **Adapter Merging & Rigorous Benchmarking**: Using `merge_and_unload()`, running the Week 3 Evaluation Harness to prove statistically significant accuracy gains over base `Qwen3-4B` and zero-shot prompting
- [ ] **Week 4 FDE Deliverable**: Versioned adapter & merged weights pushed to **Hugging Face Hub** with a comprehensive Model Card and Before/After Eval Scorecard

### Week 5 (25 hrs): Modern Quantization (`GGUF`, `llm-compressor`), High-Throughput Serving (`vLLM`) & TCO Math
**Canonical References**: `llama.cpp` docs, `vLLM` PagedAttention paper & docs, `vllm-project/llm-compressor` docs.
- [ ] **Post-Training Quantization (PTQ) Internals**: FP16/BF16 vs. INT8 vs. INT4/INT3 (`Q3_K_L`, `Q4_K_M`, `W4A16`, `FP8`) and activation-aware calibration datasets
- [ ] **Local Metal Conversion & Serving**: Converting your fine-tuned HF checkpoint to **GGUF**, quantizing with `llama.cpp`, benchmarking on M4 Metal, and importing your custom model directly into **LM Studio**
- [ ] **Cloud High-Throughput Serving (Colab L4 GPU)**: Quantizing with **`llm-compressor`** (modern replacement for archived AutoAWQ/AutoGPTQ) and serving with **`vLLM`** (PagedAttention, Continuous Batching, Prefix Caching, and Speculative Decoding)
- [ ] **FDE / Pre-Sales CTO Sizing & TCO Model**: Benchmarking **TTFT (Time-To-First-Token)**, **ITL (Inter-Token Latency)**, and concurrent throughput (req/s), and building a **Token Economics Calculator** (Self-Hosted L4/A100 vs. Frontier APIs at 10, 100, and 1,000 concurrent users)

### Week 6 (25 hrs): Enterprise Agentic RAG, LLMOps Observability & Cloud Run / HF Spaces Capstone
**Canonical References**: Chip Huyen *AI Engineering* (RAG & Production), OpenTelemetry GenAI Semantic Conventions, `RAGAS` docs.
- [ ] **Hybrid Retrieval + Reranking Pipeline**: Semantic/structure-aware chunking, Dense Embeddings + Sparse BM25, Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking evaluated with `RAGAS` (Context Precision, Faithfulness)
- [ ] **Production Async `FastAPI` Microservice**: Lifespan resource management, SSE token streaming, Pydantic strict contracts, and a **Cost/Latency Model Router** (routing between local/open-weight models and Gemini API)
- [ ] **LLMOps Observability, Guardrails & CI Gates**: OpenTelemetry `gen_ai.*` distributed tracing with **Arize Phoenix / Langfuse**, input/output prompt-injection & PII guardrails, and a **GitHub Actions CI workflow** that blocks merges on eval regressions
- [ ] **Zero-Cost 24/7 Cloud Deployment & Executive Demo**: Multi-stage Docker container deployed to **Hugging Face Spaces (Free 16GB RAM)** and **Google Cloud Run (Always-Free CPU Tier)** with a Live Telemetry UI and a 3-minute FDE client walkthrough video

---

### Optional Weeks 7–8 Buffer (Mid–Late December 2026): FDE Interview & Pre-Sales Mastery
- [ ] **Palantir / OpenAI Style "Decomposition" (Decomp) Drills**: Scoping ambiguous enterprise problems into 4-week MVPs under live architectural constraints
- [ ] **Customer Role-Play & CISO Objection Drills**: Defending air-gapped VPC deployment, hallucination mitigation, and RAG-vs-Fine-Tuning ROI using your Weeks 1–6 benchmarks
- [ ] **Certification Prep (Optional Filter Breaker)**: Google Cloud Professional ML Engineer / Databricks GenAI Engineer Associate review
