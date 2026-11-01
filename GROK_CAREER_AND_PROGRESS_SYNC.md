# Grok Bot Context: AI Engineering, MLOps & FDE Career Sync

> **SYSTEM INSTRUCTIONS FOR GROK BOT:**
> Use this living document as your single source of truth to:
> 1. **Job Search & Role Matching**: Search for high-impact **Forward Deployed Engineer (FDE)**, **Applied AI Architect**, **AI Solutions Architect**, and **Senior AI Pre-Sales Engineer** roles at frontier AI labs (OpenAI, Anthropic, Cohere, Mistral), AI/Data platforms (Google Cloud, Databricks, Palantir, Snowflake, Scale AI, NVIDIA), and enterprise GCCs/tech leaders.
> 2. **Resume & LinkedIn Tailoring**: Match the candidate's 10 years of software/infrastructure experience + the completed weekly AI engineering artifacts below to specific job descriptions.
> 3. **Mock Interview & "Decomp" Drills**: Quiz the candidate using the **Weekly FDE Customer Objections & Architecture Trade-Offs** logged in Section 4.

---

## 1. Candidate Profile Summary

- **Experience**: 10 Years in Software Engineering & Cloud/Infrastructure Architecture.
- **Hugging Face Hub Portfolio**: [`https://huggingface.co/imhlg`](https://huggingface.co/imhlg) (`imhlg`)
- **Current Transition**: Intensive 6-Week (150-Hour, 25 hrs/week) AI Systems, MLOps & Forward Deployed Engineering Sprint (Target completion: Mid-December 2026).
- **Core Value Proposition**: Unlike junior ML graduates who only know notebook wrappers, or traditional infra engineers who treat LLMs as black boxes, this candidate combines **10 years of production distributed systems/infra experience** with **deep, hardware-level LLM engineering** (tokenization, KV-cache math, LoRA/QLoRA/DPO/GRPO fine-tuning, W4A16/FP8/GGUF quantization, `vLLM` continuous batching, custom `MCP` servers, eval-driven CI/CD, and Hybrid Agentic RAG on GCP).

### Hardware & Production Stack in Active Use
- **Local Edge/Apple Silicon Stack**: Apple M4 (24GB Unified Memory), PyTorch `mps`, Apple `mlx-lm`, `llama.cpp` (Metal), LM Studio serving `Qwen3-Coder-30B-A3B` (`Q3_K_L` GGUF, MoE 3B active params).
- **Cloud GPU Stack**: NVIDIA T4 & L4 GPUs (Ada Lovelace SM 8.9, `bfloat16` / `fp8`) on Google Colab Pro (`CUDA`, `FlashAttention-2` / `SDPA`, `bitsandbytes` NF4, `llm-compressor`, `vLLM`).
- **Enterprise Cloud & Agent Stack**: Hugging Face (`transformers` v5, `datasets`, `peft`, `trl` v1+), Model Context Protocol (`FastMCP`), Evaluation Harnesses (`promptfoo`, `DeepEval`, `RAGAS`), LLMOps Observability (`OpenTelemetry` `gen_ai.*`, `Arize Phoenix` / `Langfuse`), `FastAPI` (Async SSE), Docker, **Google Cloud Run**, and **Google Gemini API / Vertex AI**.

---

## 2. Live 6-Week Sprint Progress Matrix

| Week | Module Focus | Status | Key Technologies & Keywords Unlocked |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Tokenization Internals, `transformers` v5 Inference & KV-Cache** | 🟡 **In Progress** | `Qwen/Qwen3-4B`, `Gemma-3/4`, `AutoTokenizer`, Byte-Level BPE, `apply_chat_template` (`enable_thinking`, `tools=`), Left-Padding Attention Masks, `DynamicCache`/`StaticCache`, `LogitsProcessor`, `torch.mps` / `torch.cuda` telemetry |
| **Week 2** | **Constrained Decoding, Agentic Tool Calling & Custom MCP Servers** | ⚪ Planned | Pydantic/JSON Grammar-Constrained Decoding (`outlines`), ReAct State Machines (`LangGraph`/`Google ADK`), **Model Context Protocol (`FastMCP`)**, RBAC/Least-Privilege Agent Security |
| **Week 3** | **Eval-Driven Development & Local PEFT / LoRA Fine-Tuning** | ⚪ Planned | Golden Datasets, **LLM-as-a-Judge** (`promptfoo`/`DeepEval`), Hugging Face `datasets`, LoRA Rank/Alpha VRAM Math, Apple `mlx-lm` 4-bit LoRA, `peft` + `trl` `SFTConfig` (`completion_only_loss`) |
| **Week 4** | **Cloud QLoRA (`NF4`), Preference Tuning (`DPO`/`GRPO`) & HF Hub** | ⚪ Planned | `bitsandbytes` 4-bit NormalFloat (`nf4`), Double Quantization, Paged AdamW, **Direct Preference Optimization (`DPOConfig`)**, **GRPO (`GRPOConfig`)** with verifiable rewards, Adapter Merging (`merge_and_unload`) |
| **Week 5** | **Quantization (`GGUF`, `llm-compressor`), `vLLM` Serving & TCO Math** | ⚪ Planned | Post-Training Quantization (`Q3_K_L`, `Q4_K_M`, `W4A16`, `FP8`), `llama.cpp` Metal, **`llm-compressor`**, **`vLLM`** (PagedAttention, Continuous Batching, Prefix Caching, Speculative Decoding), TTFT/ITL Benchmarking, **CTO TCO & GPU Sizing Calculator** |
| **Week 6** | **Enterprise Hybrid Agentic RAG, Observability & Cloud Run Capstone** | ⚪ Planned | Semantic Chunking, Hybrid Search (Dense + BM25 + RRF + Cross-Encoder Reranker), `RAGAS`, Async `FastAPI` SSE Streaming, Cost/Latency Model Router, **OpenTelemetry GenAI Tracing (`Arize Phoenix`/`Langfuse`)**, GitHub Actions Eval CI, **GCP Cloud Run** & **HF Spaces** |

---

## 3. Portfolio Deliverables & Resume Bullet Bank
*(Updated automatically at the end of each week with verified benchmark numbers)*

### Deliverable 1 (Weeks 1–2): Hardware-Instrumented Inference, Constrained Decoding & Enterprise MCP Server
- *Draft Resume Bullet*: Architected a hardware-instrumented local & cloud LLM inference pipeline (`Qwen3-4B` / `Qwen3-Coder-30B-A3B`) across Apple Silicon Metal (`mps`/`llama.cpp`) and NVIDIA CUDA (`bfloat16`/`SDPA`), integrating grammar-constrained JSON decoding and a custom Model Context Protocol (`FastMCP`) server with strict schema validation.
- *Key Metrics to Fill After Week 1–2 Run*:
  - M4 MPS `Qwen3-4B` (`bfloat16`) Active VRAM: `~8.0 GB` | Throughput: `[TBD] tok/s`
  - Local LM Studio `Qwen3-Coder-30B-A3B` (`Q3_K_L` MoE) VRAM: `~14.5 GB` | Throughput: `[TBD] tok/s`
  - Colab NVIDIA GPU (`T4`/`L4`) Throughput: `[TBD] tok/s`

---

## 4. Weekly FDE / Pre-Sales Interview Prep Prompts for Grok

**Grok — Drill me on these real-world FDE / Solutions Architect scenarios based on my current week:**

### Week 1 Scenarios (Active Now)
1. **The "Batched Generation Bug" Debugging Drill**:  
   *"A client's engineering team deployed an open-weight Decoder-only LLM (`Qwen3` / `Llama`) in production and noticed that single requests work fine, but batched requests output garbage tokens. Why does right-padding break autoregressive generation, and how do `padding_side='left'` and the `attention_mask` fix the self-attention matrix?"*
2. **The "CISO Air-Gapped Sizing" Pre-Sales Drill**:  
   *"A healthcare client's CISO refuses to send internal tickets to an external API and asks how much GPU memory is required to self-host a 30B parameter coding/triage assistant vs. a 4B parameter model. Walk through the exact VRAM math for FP16/BF16 vs. `Q3_K_L` quantization, Mixture-of-Experts (`30B-A3B` active parameters), and KV-cache (`DynamicCache`) scaling."*
3. **The "Reasoning Latency Trade-Off" Drill**:  
   *"When using hybrid reasoning models like `Qwen3` (`enable_thinking=True` vs `False`), how do `<think>...</think>` tokens impact Time-To-First-Useful-Token (TTFT) and token budget in customer-facing synchronous APIs, and when should you disable thinking mode in production?"*

---

## 5. Daily Progress Log (Synced In-Place to Google Drive)

- **2026-11-01 (Week 1, Day 2)**:
  - **Completed Hand-Written Byte-Level BPE Kata (`src/week1_bpe_kata.py`)**:
    - Hand-wrote the 3 core algorithmic functions (`get_pair_counts`, greedy non-overlapping `merge_pair`, and iterative `train_bpe`) without AI autocomplete.
    - Built a robust base UTF-8 byte vocabulary (0..255) to guarantee zero `<unk>` or out-of-vocabulary crashes on unseen characters and multi-byte emojis (`🚀🔥`).
    - Verified lossless round-trip decoding (`decode(encode(text)) == text`) and achieved **1.36x compression** (202 raw UTF-8 bytes $\rightarrow$ 148 BPE tokens) across 20 learned subword merges on realistic Kubernetes telemetry logs.
    - Passed all **4/4 verification tests** in the deterministic test harness.
  - **Repository Initialization**: Initialized Git repository `applied-ai-foundry` with comprehensive architecture diagram, 6-week curriculum matrix, hardware specifications, and FDE career framing in `README.md`.
  - **Up Next (Week 1, Day 3)**: Hardware-accelerated inference & KV-Cache mechanics (`src/week1_mps_starter.py` on Apple M4 `mps` with `Qwen/Qwen3-4B` in `bfloat16`, inspecting `DynamicCache` memory growth and writing custom `LogitsProcessor`).

- **2026-10-31 (Week 1, Day 1)**:
  - **Environment & Stack Locked**: Configured local `.venv` (`requirements.txt`) on Apple M4 (24GB Unified Memory) with `torch==2.14.1` (`mps` verified), `transformers==5.20.2` (v5 API), `tokenizers==0.23.3`, and `accelerate==1.15.0`.
  - **Completed Lab (`src/week1_tokenizer_deep_dive.py` on `Qwen/Qwen3-4B`)**:
    - Inspected Byte-Level BPE vocabulary (`151,643` base tokens, `151,669` total; `eos='<|im_end|>'` `151645`, `pad='<|endoftext|>'` `151643`), CamelCase K8s log subword splits, and single-digit numeric tokenization (`984710293847` $\rightarrow$ 12 individual digit tokens).
    - Verified `apply_chat_template` reasoning control (`enable_thinking=False` pre-filling `<think>\n\n</think>\n\n` for zero-thinking TTFT vs. `enable_thinking=True`) and native Python tool schema serialization into `<tools>...</tools>` and `<tool_call>` XML tags.
    - Verified Decoder-only batched padding (`tokenizer.padding_side = "left"`) and attention masking (`0` for left-padded `<|endoftext|>` tokens, `1` for active prompt tokens).
  - **Up Next (Week 1, Day 2)**: Complete hand-written Byte-Level BPE kata (`src/week1_bpe_kata.py`), pass Explain-Back Gate, and benchmark `src/week1_mps_starter.py` (`Qwen/Qwen3-4B` in `bfloat16` with `DynamicCache` on M4 Metal).
