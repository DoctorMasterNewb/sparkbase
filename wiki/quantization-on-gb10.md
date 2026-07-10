# Quantization on GB10

> **area:** quantization
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-m3-vision, S-nemotron-rpc, S-diffusiongemma, S-forum-fp4psa, S-forum-mxfp4-patches, S-forum-nvfp4-ray, S-forum-nvfp4-100b, S-forum-kvarn, S-forum-spark-auto-round, S-forum-kv-bench-llamacpp, S-forum-turboquant, S-forum-stream-loading, S-forum-nvfp4-quant-gp10, S-forum-vllm-019-vs-023, S-forum-qwen36-27b-fp8, S-forum-qwen122-nvfp4-quant, S-forum-nvfp4-mistral-3node, S-forum-flux2-nvfp4-compute
> **updated:** 2026-07-10

GB10 has **no native FP4 compute and no native FP8 block-scale**. That one fact decides which quant
to pick. Decode is **bandwidth-bound** (`[[wiki/platform-gb10.md]]`), so the winning quant is usually
the one that moves the **fewest weight bytes/token** — even if its kernel is a slow weight-only
decompress, because at low batch you're memory-bound, not compute-bound.

## What runs natively vs. falls back to Marlin

| Format | GB10 path | Speed | Notes |
|---|---|---|---|
| **FP8 online dynamic** (`--quantization fp8` on bf16 weights) | native **CUTLASS** (`CutlassFP8ScaledMMLinearKernel`) | **fast** | quantizes at load, no pre-quant checkpoint. The genuine FP8 fast path. ~2.08× bf16 decode. |
| **FP8 block-scale checkpoint** (compressed-tensors `float-quantized`, DeepSeek-style) | **Marlin** weight-only ("no native support for FP8") | slow-ish | NOT the fast path. This is why "FP8 = fast on GB10" fails for block-scaled checkpoints (e.g. Holo FP8). |
| **NVFP4 — ModelOpt** (`quant_method: modelopt`, `hf_quant_config.json`) | **Marlin FP4 MoE** + FlashInfer FP8 linear (or FLASHINFER_CUTLASS) | fastest for MoE | the only NVFP4 that works. Fewest bytes ⇒ wins despite Marlin decompress. |
| **NVFP4 — compressed-tensors** (`nvfp4-pack`, `.weight_packed`) | **broken** | — | garbage / won't load correctly on GB10. Avoid. Confirmed multiple times. |
| **MXFP8** (mixed-precision, UE8M0 scales) | FlashInfer-CUTLASS MXFP8 | works | needs explicit dispatch in some vLLM builds (see MiMo mods). Don't reciprocal-invert `weight_scale_inv`. |
| **AWQ 4-bit** | Marlin | works, **fast decode** | MiniMax-M2.x default. For MiniMax-M2.7, AWQ **decodes ~1.4× faster than NVFP4** single-stream (~24 vs ~16.5 tok/s) — same ~4-bit, so it's kernel efficiency not bytes. Measure; don't assume NVFP4 wins. |
| **AutoRound (mixed 2/3/4/8-bit)** | plugin-dependent | works w/ plugin | needs OneCompression `autoround_mixed` for 16-bit-base mixed checkpoints (see MiniMax-M3). |
| **GGUF Q8_0 / Qx** | llama.cpp | works | the llama.cpp path; see `[[wiki/llama-cpp-rpc.md]]`. |

## Rules of thumb

- **[proven]** **New model, want speed:** prefer a **ModelOpt NVFP4** checkpoint if one exists; else
  **online dynamic FP8** (`--quantization fp8`) on bf16 weights. Avoid block-scaled FP8 checkpoints
  unless that's all there is. **But if an AWQ checkpoint also exists, measure both** — for MiniMax-M2.7
  AWQ *decoded faster* than NVFP4 (kernel efficiency); the "prefer NVFP4" rule is a default, not a law.
- **[proven]** **Probe before you download.** Fetch `config.json` over HTTP and read
  `quantization_config.quant_method` / `.format`: `modelopt` (good) vs `compressed-tensors`
  block-scale (Marlin fallback) vs `nvfp4-pack` (broken). Saves a 20–35 GB mistake.
- **[proven]** **MoE + NVFP4 is the GB10 sweet spot.** Few active params/token × few bytes/param.
  Qwen3.6-35B-A3B NVFP4 hit ~142 tok/s single-stream; Holo-35B-A3B NVFP4 ~77 tok/s. A *dense* model of
  similar size is far slower (activates all params) — Qwen3.6-27B dense FP8 ~30 tok/s.
- **[proven]** **NVFP4 is compute-bound at high batch.** Because the FP4 MoE is Marlin weight-only
  decompress, aggregate throughput plateaus (Holo NVFP4 saturated ~900 tok/s at ~96–128 concurrency). On
  native-FP4 datacenter Blackwell it would scale further; Spark caps here.
- **[proven]** **Runtime bf16→low-bit quant can OOM the KV cache.** It transiently holds both the bf16
  and quant copies, starving the KV pool to "0 concurrent sequences." Use a **pre-quantized** checkpoint.
- **[proven]** **KV-cache dtype:** `fp8` (e4m3) KV is the common choice and is plenty precise; sparse-
  attention/hybrid archs make KV cheap regardless (see `[[wiki/attention-and-kv-cache.md]]`).

## Quant-related loader bugs (GB10-seen)

- **[proven]** **Fused QKV blind-chunk corruption (NVFP4 MoE).** Checkpoints store fused QKV as canonical
  `[Q_all|K_all|V_all]`; vLLM's blind `chunk(tp_size, dim=0)` is shape-correct but **corrupts K/V
  rows** → multilingual garbage. Fix: load via `QKVParallelLinear.weight_loader` (MiMo
  `fix-mimo-v2-vllm` mod). See `[[wiki/models/mimo-v2.5.md]]`.
- **[proven]** **Missing `packed_modules_mapping` on Omni wrappers** → fused `gate_up_proj` resolves to
  Unquantized, MXFP8 bytes read as bf16 → gross garbage. Add the mapping to the Omni class.
- **[proven]** **MXFP8 not dispatched.** Some vLLM builds dispatch FP8/NVFP4/W4A16 but not MXFP8 → MXFP8
  layers silently fall to Unquantized. Add MXFP8 to `ModelOptMixedPrecisionConfig.get_quant_method`.
- **[proven]** **Old MoE loaders miss NVFP4 per-expert scales** → `KeyError: '…experts.w2_weight_scale'`.
  Newer vLLM image fixes it (see `[[wiki/containers-and-tooling.md]]`).
- **[proven]** **NVFP4 MoE default backend can't pad some experts.** Symptom: engine init dies with
  `NotImplementedError: Intermediate size padding for w1 and w3, for FLASHINFER_CUTLASS NvFp4 backend,
  but this is not currently supported`. The default NvFp4 MoE backend (`FLASHINFER_CUTLASS`) can't pad
  the expert intermediate size for some checkpoints (hit on DiffusionGemma-26B-A4B-NVFP4). Fix: force
  Marlin with **`VLLM_TEST_FORCE_FP8_MARLIN=1`** (→ `Using 'MARLIN' NvFp4 MoE backend`). ⚠ On our vLLM
  build 0.23.1rc1.dev520, **`VLLM_NVFP4_GEMM_BACKEND` does not exist** ("Unknown vLLM environment
  variable", no-op) and neither does `VLLM_USE_FLASHINFER_MOE_FP4` (only `..._INT4`) — some serving-
  supervisor config docs list stale var names for this build; `VLLM_TEST_FORCE_FP8_MARLIN` is the working
  lever. See `[[wiki/models/diffusiongemma.md]]`.
- **[proven]** **AutoRound 16-bit-base reject:** stock validator only accepts bits ∈ {2,3,4,8}; a 16-bit
  base with per-module overrides needs the `autoround_mixed` method + a `config.json` `quant_method`
  rename.

## See also
`[[wiki/platform-gb10.md]]` · `[[wiki/attention-and-kv-cache.md]]` · `[[wiki/containers-and-tooling.md]]` · per-model pages under `wiki/models/`

## Forum ingest: AWQ vs NVFP4 decode (2026-07-08)

- **[reported]** **AWQ 4-bit outperforms NVFP4 on decode by ~32%** on GB10 — confirmed across multiple
  models on both single-node and 2-node clusters. Qwen3-VL-235B-A22B AWQ 4-bit vs NVFP4: AWQ ~32%
  faster single-concurrency, ~18% faster at c=10, snappier TTFT (S-forum-fp4psa, eugr). This
  corroborates the first-party MiniMax-M2.7 finding (AWQ ~24 vs NVFP4 ~16.5 tok/s) and generalizes it
  beyond MiniMax. FP8 and AWQ 8-bit perform roughly equally (FP8 slightly faster on prompt processing,
  AWQ 8-bit slightly faster on token gen).

## Forum ingest: MXFP4 online quantization patches (2026-07-08)

- **[conjecture]** **MXFP4 online quantization patches for SM121** (S-forum-mxfp4-patches,
  amasawa_seiji): a set of vLLM 0.17.0 patches enable BF16→MXFP4 online quant for MoE experts,
  attention (qkv_proj, o_proj), and lm_head via Marlin backend, plus SM121 CUTLASS MoE fixes and
  GDN Triton kernel fix for Qwen3.5. Reported results (2-node TP=2, 1024 in / 128 out):
  - Qwen3.5-35B-A3B: vanilla 42.85 → patched **70.68 tok/s** (+65%)
  - gpt-oss-120b: vanilla 51.82 → patched **80.88 tok/s** (+56%)
  - Enables TP=1 single-GPU inference for MXFP4 (vanilla vLLM cannot do this on GB10).
  - GitHub: `github.com/namake-taro/vllm-custom`
  - **[conjecture]** The SFA/SFB bug in CUTLASS for SM121 is still unmerged upstream.

## Forum ingest: NVFP4 CUTLASS failure on sm_121 (2026-07-08)

- **[conjecture]** FlashInfer detects GB10 is not SM100 (B200), falls back to CUTLASS FP4 — but CUTLASS
  FP4 **also fails** on sm_121 with `RuntimeError: [FP4 gemm Runner] Failed to run cutlass FP4 gemm on
  sm120. Error: Error Internal` (S-forum-nvfp4-ray). This was on vLLM 0.12.0 with a ptxas symlink fix.
  **[reported]** Triton's bundled ptxas 12.8 does NOT support sm_121a; symlink to CUDA's ptxas
  (`ln -sf /usr/local/cuda/bin/ptxas …/triton/backends/nvidia/bin/ptxas`) to fix Triton compilation.
  This is a known toolchain gap on GB10.

## Forum ingest: Distributed NVFP4 quantization on 2x Spark (2026-07-08)

- **[conjecture]** Single-node `modelopt hf_ptq.py` **silently OOM-kills** on Spark for 100B+ models
  (S-forum-nvfp4-100b). `accelerate.infer_auto_device_map` misdetects GB10 unified memory as a ~5.2 TB
  GPU → silent OOM in single-node flows. Fix: distributed Ray pipeline, layer-sharded across both
  Sparks, BF16 calibration → NVFP4 export via modelopt 0.43.
- **[conjecture]** modelopt 0.43 NVFP4 export writes `input_activations.dynamic=false` but does NOT
  emit `input_scale` keys → vLLM registers uninitialized Parameters → garbage output until
  `input_scale=1.0` sidecar keys are injected for every quantized Linear (840 for a 105B model).
  Six-fix list documented in the model card (Kaleto/Anubis-Pro-105B-NVFP4).

## Forum ingest: KVarN, Spark Auto Round, KV benchmarks, TurboQuant (2026-07-09)

- **[conjecture]** **KVarN: native vLLM KV-cache quantization backend** (S-forum-kvarn, Huawei):
  calibration-free, one-flag KV-cache quantization delivering 3-5× more KV capacity and up to ~1.3×
  throughput of FP16, with FP16-level accuracy. Up to ~2.4× TurboQuant throughput. **[conjecture]**
  Does NOT work with Qwen 3.6 currently — the "no model changes" claim is not fully accurate. GitHub:
  `huawei-csl/KVarN`. Used in the MiniMax-M3-W4A16-GPTQ recipe (S-forum-m3-w4a16-gptq) to reach 262K+
  context on 2× GB10.
- **[conjecture]** **Spark Auto Round** (S-forum-spark-auto-round, whpthomas): Int4 AutoRound
  quantization tool focused on GB10, using an OpenCode Instruct calibration dataset. Sensitivity-aware:
  detects problematic layers via cosine similarity (perplexity) < 0.99 and PSNR < 45 dB, keeps sensitive
  layers in FP16 while quantizing insensitive ones. Hypothesis: replacing sensitive layers with FP16
  improves quality. Qwen3.6-35B-A3B needs an attribute pattern match fix. Shared expert gate layers
  (not divisible by 32) are kept in full precision (bf16) — expected, negligible impact.
- **[conjecture]** **KV cache quantization on llama.cpp** (S-forum-kv-bench-llamacpp, nmaine): on
  DGX Spark with Nemotron-3-Nano-30B (Q4_K_XL), 128K context:
  - `q4_0` KV is **92% slower at 64K context** — dequantization overhead destroys prompt processing.
  - `q4_0` KV uses **MORE memory than f16** — scale/zero-point metadata overhead exceeds compression
    savings on Spark's unified memory architecture.
  - `q8_0` is the only worthwhile KV quantization — 2× compression, <5% speed hit at all context lengths.
  - f16 is fine for most workloads: at 64K tokens, KV cache is under 2 GB out of 128 GB available.
  - Recommendations: <16K ctx → f16; 16–64K → `q8_0`; 64K+ → wait for TurboQuant or NVFP4.
- **[conjecture]** **TurboQuant KV cache** (S-forum-turboquant, bjk110): vLLM 0.19.1 patched with
  PR #38280 on 2× GB10. Increased KV cache capacity from 155K to 413K tokens. Gather-free Triton
  decode reads paged cache directly (no full memcpy). CUDA WPH (warp-per-head) decode for SM121/aarch64,
  BLOCK_D=128/256. WPH path slightly faster than Triton on Qwen3.5 at c=2 and c=4. Issues: page-size
  mismatch with Mamba/GDN layers (fix: `_next_pow2()` padding), CUDA graph capture crash (fix:
  capture-aware branching for `.tolist()`/`.unique()` CPU ops).
- **[conjecture]** **STREAM LOADING** (S-forum-stream-loading, amasawa_seiji): custom vLLM 0.17.1 build
  that reads only necessary expert/layer chunks from storage, quantizes to 4-bit on-the-fly, and places
  result on GPU — eliminates the need to hold both BF16 and 4-bit data simultaneously. Confirmed running
  BF16/FP8 models on single Spark: Qwen3.5-397B-A17B-FP8 (~96.7 GiB/GPU TP=2), Nemotron3-120B-BF16,
  Qwen3.5-122B-A10B. Also includes NF4 sub-mode (normal-distribution-based 16-level partition, better
  quality than pure MXFP4 E2M1) and automatic KV cache allocation (no manual `--gpu-memory-utilization`).
- **[conjecture]** **ModelOpt NVFP4 quantization on Spark is CPU-bound** (S-forum-nvfp4-quant-gp10):
  the `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` container running ModelOpt's
  `huggingface_example.sh --quant nvfp4` shows **zero GPU load, all CPU**. Multiple users confirm
  the same behavior — the quantization calibration phase doesn't use the GPU on GB10. The process
  may fail silently for some models.
- **[conjecture]** **vLLM 0.19 → 0.23 performance regression** (S-forum-vllm-019-vs-023): Qwen3.5-122B
  AutoRound on same Spark: vLLM 0.19 = 37 tok/s (Q&A), 41.2 tok/s (code); vLLM 0.23 = 32.5 tok/s (Q&A),
  36.9 tok/s (code). Memory footprint also regressed: 104 GB → 120 GB unified RAM after load. ~12%
  speed regression + 15% memory regression across versions. Tag existing images before upgrading.
- **[conjecture]** **Dense model MTP math** (S-forum-qwen36-27b-fp8): Qwen3.6-27B FP8 at 270 GB/s
  bandwidth: 27B × 1 byte = ~27 GB/pass → ~10 tok/s theoretical ceiling (measured 7.8 baseline). MTP
  nst=3 → 15.2 tok/s (1.94×). NVFP4 would move ~7 GB/pass → ~38 tok/s theoretical. MTP is a "free"
  speedup for bandwidth-bound dense models — the decode ceiling is pure bandwidth math.
- **[conjecture]** **Heterogeneous NVFP4 quantization** (S-forum-nvfp4-mistral-3node): NVFP4
  quantization of a 123B Mistral-Large finetune on a 3-node heterogeneous cluster (2× Spark + 1× RTX
  3090) via Ray. NVFP4 calibration math runs in BF16 (not FP4), so Ampere GPUs participate as normal
  Ray actors — exported model is byte-identical to an all-Blackwell cluster output. Cross-node Ray RPC
  over 2.5 GbE adds ~50 sec total to a 256-sample calibration (compute-side dominates, network never
  the bottleneck). Three extra fixes on top of the 6 already documented for 100B+ models.

## Forum ingest: FLUX.2 NVFP4 W4A4 compute (2026-07-10)

- **[conjecture]** **NVFP4 W4A4 (activation-quantized) gives ~3× speedup on FLUX.2-dev** on DGX Spark
  (S-forum-flux2-nvfp4-compute, vr8vr8): torchao on-the-fly quantization with W4A4 (weights AND
  activations quantized to FP4) runs the matmul in actual FP4 on Blackwell tensor cores via Triton
  kernels. FLUX.2-dev 28 steps @ 1024²: BF16 ~2.3 min → NVFP4 ~45 s (~3× text-to-image, ~2.3× edit).
  VRAM: ~112 GB BF16 → ~66 GB NVFP4 (~40% less). After one-time quant, subsequent boots load the
  quantized weights directly at ~66 GB with no BF16 spike.
  - **Critical distinction:** most quantized FLUX files floating around (fp8-mixed, gguf, naively-loaded
    "nvfp4" checkpoints) are **weight-only** — weights stored small but matmul upcasts to BF16. Memory
    saving but little-to-no speedup (sometimes slower). torchao W4A4 is what gets the real FP4 compute.
  - **[conjecture]** `modelopt_fp4` / prequantized-NVFP4-checkpoint path hits a diffusers
    unpack/shape bug on sm_121a — on-the-fly torchao is the working route. Same "don't pass
    modelopt_fp4 on SM121A" reported by others.
  - `mslk` is a missing dependency — without it torchao errors "mslk is required for NVFP4 triton
    quantization." CUDA 13 + Blackwell (sm_120a/121a) required for the FP4 Triton kernels.
  - PR to eugr/spark-vllm-docker (#313) — first image-generation model in the repo (headless
    OpenAI Images-API server, not GUI/ComfyUI).
