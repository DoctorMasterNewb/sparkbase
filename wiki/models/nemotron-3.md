# Nemotron-3

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-nemotron-rpc, S-swapper, S-forum-nemotron-super-mtp, S-forum-nemotron-ultra-4x, S-forum-nemotron-super-abi
> **updated:** 2026-07-09

NVIDIA Nemotron-3 — **hybrid Mamba-2 + attention MoE** (`nemotron_h_moe`). Most layers are SSM with a
few attention layers (2 KV heads), so KV is cheap and native context is huge. Two paths on GB10.

## Nemotron-3-Super-120B-A12B (Q8 GGUF) — llama.cpp RPC, 2-node

- `timteh673/Nemotron-3-Super-120B-A12B-Uncensored-GGUF` Q8_0 (**128 GB single blob** > 121 GB/node →
  must split). Engine: **croll83 `llama.cpp-dgx` fork** (sm_121a, registers `nemotron_h_moe`).
- **[proven]** 2-node **pipeline RPC** (`[[wiki/llama-cpp-rpc.md]]`): ~61 GB/node, **~10.5 tok/s**
  decode, coherent. Native ctx **1,048,576** (no YaRN) — ran 1M × 4 slots, ~43 GB free/node (KV cheap:
  Mamba-2 hybrid).
- **[proven]** **`--no-mmap` REQUIRED** (else unified-mem OOM-kill mid-load). RPC has no auth — fabric
  IP only. If decode garbles with the hybrid SSM, bias `--tensor-split` so all SSM layers stay on one
  node.
- **[proven]** **Gate on arch BEFORE the 128 GB download:** read `general.architecture` via a 2 MB
  HTTP range read.

## Nemotron-3-Nano-Omni (vision/omni) — the worker vision/omni unit

- **[proven]** `AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16` — 30B NemotronH hybrid
  (Mamba2 + attn + 128-expert MoE), BF16, omni (text + image RADIO + audio Parakeet), ctx 200k. Image
  `ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1`. Single-node on the worker. (Also NVFP4
  variants exist via Atlas: `nemotron-3-nano-nvfp4`, `nemotron-3-super-nvfp4`.)

## See also
`[[wiki/llama-cpp-rpc.md]]` · `[[wiki/attention-and-kv-cache.md]]` (hybrid SSM = cheap KV) · `[[wiki/engines.md]]`

## Forum ingest: MTP on Super-120B, Ultra-550B on 4× Spark, ABI fix (2026-07-09)

- **[reported]** **Nemotron-3-Super-120B MTP works on 4× Spark SGLang** (S-forum-nemotron-super-mtp,
  ht12): the published state was "crashes" (vLLM t/366660) or "0% draft acceptance, accept_len=1.00"
  (SGLang sglang#21138). On a build carrying June-2026 NemotronH-MTP fixes (SGLang 0.5.13-dev,
  image `xomoxcc/dgx-spark-sglang:0.5.13-dev-nemotronh-mtp-sm121`), MTP delivers **accept_len ≈2.7,
  1.70× single-stream, 1.37× at n8 concurrency**. The 3-step/4-draft depth beats NVIDIA's own cookbook
  5/5 recipe. Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (~67 GB NVFP4 weights). NCCL
  2.30.4 required.
  - **[conjecture]** **Single-node limitation:** the 67 GB weights fit one 128 GB Spark, but the
    concurrency limit isn't KV — it's the **Mamba state pool**: `max_running_requests =
    max_mamba_cache_size // per-request-slots`. One Spark leaves ~37.5 GB free; the 96-slot Mamba pool
    wants ~27 GB on top of KV + CUDA graphs → doesn't fit. Must shrink Mamba pool (96→24) or use TP=4.
    The engine's "increase --mem-fraction-static" advice is a red herring — avail mem is physically
    pinned at ~37.5 GB regardless of the fraction.
- **[reported]** **Nemotron-3-Ultra-550B on 4× Spark SGLang** (S-forum-nemotron-ultra-4x, ht12):
  `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` boots on mainline `scitrera/dgx-spark-sglang:0.5.12`
  (no custom build). TP=4, EP=4, RoCE. **~42–43 tok/s n8 peak** (5.3 tok/s/request @ n8), 512K context.
  Model is `NemotronHForCausalLM` — Mamba2 + MoE + attention hybrid: 108 layers (48 mamba / 48 moe /
  12 attention), 550B total / 55B active LatentMoE, 512 routed + 1 shared experts, NoPE. Quant is
  `modelopt_mixed` (FP4 expert FFN @ group_size 16, FP8/BF16 for attention/latent/embeddings).
  Weights land at 83.7 GB/GPU (less than naive ~107 GB estimate — mixed-precision tensors are smaller).
- **[conjecture]** **Nemotron-3-Super NVFP4 via vLLM TP=2 — ABI fix** (S-forum-nemotron-super-abi):
  24 tok/s on 2× Spark. The `c10::MessageLogger` crash (`_ZN3c1013MessageLoggerC1E...`) is a
  **cu130/cu132 ABI mismatch**: the prebuilt vLLM wheel is `cu132` but the Dockerfile installs PyTorch
  from `cu130` index → different `libc10.so` ABI. Fix: change `cu130` → `cu132` in the Dockerfile
  (lines 48 + 259). Also resolved by newer `.dev176` prebuilt wheel (clear wheel cache to pull).
