# Quantization on GB10

> **area:** quantization
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-m3-vision, S-nemotron-rpc, S-diffusiongemma, S-forum-fp4psa, S-forum-mxfp4-patches, S-forum-nvfp4-ray, S-forum-nvfp4-100b, S-forum-kvarn, S-forum-spark-auto-round, S-forum-kv-bench-llamacpp, S-forum-turboquant, S-forum-stream-loading, S-forum-nvfp4-quant-gp10, S-forum-vllm-019-vs-023, S-forum-qwen36-27b-fp8, S-forum-qwen122-nvfp4-quant, S-forum-nvfp4-mistral-3node, S-forum-flux2-nvfp4-compute, S-forum-nvfp4-worth, S-forum-unsloth-qwen36, S-forum-nvfp4-broken, S-forum-glm52-8x, S-forum-gridbook
> **updated:** 2026-07-25

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

## Forum ingest: Unsloth NVFP4 quants on GB10, flashinfer_b12x gap (2026-07-15)

- **[reported]** **Unsloth NVFP4 quants are ~15% slower than nvidia NVFP4 on GB10** (S-forum-unsloth-qwen36):
  3 independent forum benchmarks agree — hedelyuk.alexandr (−15.2% avg across 4 workloads), J-R (−17%
  at c=1), TheAwakenOne (75→97 tok/s tuned, still behind nvidia). All on single-node DGX Spark, vLLM 0.24.
  Unsloth's "2.5× faster" claim is B200-specific and does not transfer to sm_121. The nvidia NVFP4 Marlin
  MoE path remains the fastest NVFP4 option on GB10.
- **[conjecture]** **`flashinfer_b12x` kernel unavailable on stock vLLM** despite `has_flashinfer_b12x_gemm()`
  returning True on sm_121 (jbourny): `--moe-backend flashinfer_b12x` → `ValueError: no 'flashinfer_b12x'
  kernel exists for this layer type`. Unsloth's optimum recipe requires b12x but falls back to Marlin on
  stock builds. JW2026 measured <3% difference with b12x flags (within margin of error). The b12x path
  may require a custom vLLM build or specific container image.
- **[conjecture]** **Unsloth "speedup" may be W4A16 bypass** (robert287): Unsloth's performance boost on
  Qwen may come from bypassing NVFP4 on weights via W4A16 — consistent with GB10 findings that AWQ/W4A16
  Marlin and NVFP4 Marlin are within noise on decode (kernel efficiency, not bytes, is the differentiator
  at ~4-bit — see AWQ-vs-NVFP4 section above).
- **[reported]** **Quality parity**: tool-eval-bench scores equal to nvidia NVFP4 with more run-to-run
  variance (azampatti, 2 independent eval runs).

## Forum ingest: NVFP4 meta-analysis — NVFP4 on GB10 state of play (2026-07-16)

- **[reported]** **NVFP4 leaves ~half of layers in BF16, unlike Int4 which quantizes all layers**
  (S-forum-nvfp4-broken, tenari): when a model is quantized to NVFP4, approximately half the layers
  stay in BF16 due to quality-degradation concerns. Int4 (via AutoRound) quantizes ALL layers. This
  structural difference explains part of why NVFP4 underperforms vs Int4 on decode: more BF16 layers
  = more bytes/token read from memory. The community is working on full-NVFP4 quantization (tenari's
  PrismQuant project). Corroborates the existing `[proven]` finding that fewer weight bytes/token ⇒
  faster decode — NVFP4's mixed-precision nature means it moves more bytes than a pure Int4 path.
- **[reported]** **TRT-LLM NVFP4 is slower than GGUF Q4_K_M via LM Studio on the same Spark**
  (S-forum-nvfp4-broken, DropTheBeat citing two separate forum threads): Llama-3.3-70B-Instruct-NVFP4
  on TensorRT-LLM = 5 tok/s decode (one reporter), 2.5 tok/s (another reporter). The same 70B model
  via GGUF Q4_K_M in LM Studio = 4.6–4.9 tok/s. NVIDIA's own NVFP4 model on NVIDIA's own TRT-LLM is
  slower than a non-NVIDIA quant on non-NVIDIA tooling. This corroborates the existing `[reported]`
  finding that AWQ/Int4 outperforms NVFP4 on decode on GB10 (kernel efficiency, not bytes).
- **[reported]** **Nemotron-3-Super NVFP4 bandwidth efficiency is 42–48% of theoretical ceiling**
  (S-forum-nvfp4-broken, DropTheBeat): Nemotron-3-Super-120B has 12B active params/token. At NVFP4
  (0.5 bytes/param), each decoded token reads ~6 GB of active weights. GB10 bandwidth = 273 GB/s
  (NVIDIA datasheet) → theoretical ceiling ~45 tok/s. Measured 19–22 tok/s = 42–48% of ceiling.
  A well-optimized NVFP4 path should deliver ~30–40 tok/s (60–80% efficiency, which is routine on
  GB10 in other configurations). The hardware is leaving roughly half its achievable throughput on
  the floor on NVIDIA's own NVFP4-native flagship model. This is NOT a bandwidth limitation — it's
  a software/kernel efficiency gap. Corroborates the existing `[proven]` bandwidth-bound decode
  finding and quantifies the NVFP4 kernel overhead.
- **[reported]** **NVFP4 is now operational on GB10 via community Docker** (S-forum-nvfp4-broken,
  jwarner): a completely operational NVFP4 path exists on GB10 as of the last month, even on
  inference frameworks NVIDIA doesn't control. This is the default on the community Docker image
  (spark-vllm-docker). Required PRs across many open repositories, mostly by NVIDIA employees.
  FlashInfer 0.6.8.1 has further improvements merged. This partially mitigates the "NVFP4 is broken"
  framing — the path works but performance is still sub-optimal vs theoretical ceiling.
- **[conjecture]** **FlashInfer 0.6.8.1 brings further NVFP4 improvements** (S-forum-nvfp4-broken,
  jwarner): merged just days before the post. Specific improvements not detailed but described as
  incremental. Consistent with the existing `[reported]` finding that FlashInfer-CUTLASS is now
  stable for NVFP4 recipes (S-forum-m27-recipe).

## Forum ingest: b12x W4A8 MoE, stale topk buffer, Int4-Int8 mix (2026-07-17)

- **[conjecture]** **b12x W4A8 MoE backend (lukealonso/b12x @ 97b3d64) raises GLM-5.2 decode on GB10**
  (S-forum-glm52-8x, ciprianveg): `b12x` (a unified SM120 sparse-MLA + PCIe DCP collectives backend)
  with W4A8 MoE quantization raised GLM-5.2 Int4-Int8Mix decode from ~28–49 (MTP k=4, older image)
  to 33–54 t/s on 8× GB10. The W4A8 path weights are INT4 but activations are INT8 — distinct from
  NVFP4 W4A4 (FP4 weights AND activations via Marlin decompress) and from weight-only AWQ/Marlin
  W4A16. On GB10 (no native FP4/FP8-blockscale), W4A8 activations run via the native FP8 online-
  dynamic CUTLASS path (weights INT4 Marlin-decompress, activations FP8 CUTLASS) — this is the
  "fewest bytes" principle extended to activations. Stack: vLLM v16-unified fork + b12x.
- **[conjecture]** **Stale topk_indices_buffer in flashinfer SM120 sparse MLA (PR #46994)** — a class
  of subtle bugs where `_maybe_share_lm_head` swaps the indexer's buffer but the backend keeps a
  stale reference → garbage DSA (deeply-sharded attention) output and ~30% MTP acceptance instead
  of ~85%. Two fixes needed: (1) patch 04 — flashinfer SM120 sparse MLA stale topk_indices_buffer
  (vLLM PR #46994), (2) patch 06 — same stale-buffer fix applied to b12x_mla_sparse.py (PR #46994
  Fix #4). Without both, MTP acceptance collapses silently (no error — just bad acceptance).
  This is a GB10/sm_121-specific kernel bug in the sparse-MLA path (the DSA / sparse attention
  architecture used by GLM-5.2 and DeepSeek-V4-Flash). (S-forum-glm52-8x)
- **[conjecture]** **Quantized NextN draft token mapping required for MTP with quantized drafts**
  (S-forum-glm52-8x, patch 03 from CosmicRaisins): without the quantized NextN draft token mapping
  patch, quantized drafts silently build unquantized and MTP acceptance collapses. A GB10-specific
  MTP + quant interaction — the draft model must use the same quant mapping as the main model or
  the draft tokens are computed in the wrong precision.
- **[conjecture]** **`VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1`** (S-forum-glm52-8x): treats spec-extend
  as decode so the B12X indexer path stays consistent during MTP — a b12x-specific env var for
  sparse-MLA models (GLM-5.2, DS-V4-Flash) using the b12x backend with MTP.
- **[conjecture]** **Int4-Int8 mix quant (QuantTrio/GLM-5.2-Int4-Int8Mix)** (S-forum-glm52-8x): a
  mixed-precision checkpoint where MoE experts are INT4 and other layers (attention, MTP heads)
  are INT8 — distinct from pure NVFP4 (W4A4) or pure AWQ (W4A16). On GB10 this runs via Marlin
  decompress for INT4 experts + native FP8 CUTLASS for INT8 activations. The 8× GB10 run reached
  ~1,200 t/s prefill and 33–54 t/s decode — the fastest reported GLM-5.2 result on DGX Spark
  (vs ~22 tok/s at TP=4 NVFP4/AWQ-INT4 in S-forum-glm52-4x, ~24 tok/s NVFP4 MTP4 in
  S-forum-glm52-mtp-fix). Whether the speedup is from the quant mix, the 8× scale, the v16 branch,
  or the b12x backend cannot be isolated from a single source.

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

## Forum ingest: FlashInfer-CUTLASS stable, diffusion NVFP4 weight-only vs activation-quantized (2026-07-10)

- **[reported]** **FlashInfer-CUTLASS is now stable for NVFP4 recipes** (S-forum-m27-recipe, eugr):
  autotuner exceptions fixed, minor FlashInfer optimizations merged. Eugr plans to switch all NVFP4
  recipes from `VLLM_CUTLASS` to `flashinfer-cutlass`. On MiniMax-M2.7-NVFP4 2× Spark: FlashInfer-CUTLASS
  beats CUTLASS on both latency (111 vs 122 ms) and throughput (tg128 24.12 vs 22.04). Best config:
  `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, `VLLM_USE_FLASHINFER_MOE_FP4=1`,
  `VLLM_FLASHINFER_MOE_BACKEND=throughput`, `VLLM_FLOAT32_MATMUL_PRECISION=high`.
- **[conjecture]** **NVFP4 weight-only quant gives only modest diffusion speedups** (S-forum-diffusion-speeds,
  ijontichy): weight-only NVFP4 on diffusion models (via torchao, NOT activation-quantized W4A4) gives
  ~1.1–1.4× speedup — FLUX.2-klein 4.4→3.3s, Z-Image-Turbo 7.2→5.6s, ERNIE 8.8→6.4s, Krea2 13.9→12.4s.
  This is a fraction of the ~3× from FLUX.2-dev W4A4 (activation-quantized, real FP4 compute via
  Triton — see section above). The distinction is critical: weight-only NVFP4 saves memory but the
  matmul upcasts to BF16; W4A4 runs actual FP4 compute. Diffusion model weight-only NVFP4 ≠ LLM NVFP4
  (where MoE expert dequant via CUTLASS/FlashInfer is the kernel path).

## Forum ingest: NVFP4 quant recipe, TensorRT-LLM errors (2026-07-13)

- **[conjecture]** **NVIDIA refreshed the official NVFP4 quantization recipe for DGX Spark**
  (S-forum-nvfp4-worth, paul448): NVIDIA updated the `build.nvidia.com/spark/nvfp4-quantization`
  recipe page. A community gist accompanies it for manual NVFP4 conversion. The poster attempted
  to convert Qwen3-27B and Qwen3.6-27B but hit **TensorRT-LLM errors** — not yet reproducible to a
  clean result. Focusing on distilled / ≤30B models for eval workflow. Single source; no working
  conversion numbers reported. Corroborates existing finding that NVFP4 quant on Spark is
  CPU-bound and can fail silently (S-forum-nvfp4-quant-gp10).

## Forum ingest: PrismaQuant GridBook codebook quant (2026-07-25)

- **[conjecture]** **GridBook — codebook quant with native-GB10 dequant via tensor-core lookup**
  (S-forum-gridbook, tenari/RobTand): a vLLM plugin exposing a **ladder of 41 codebook quant formats**
  spanning **1.781 to 6 bits** (quarter-bit increments + a few signed-bit experimental formats).
  Builds on PrismaQuant (the bit-allocator, S-forum-prismaquant) by adding a much finer format menu
  than the original {NVFP4, MXFP8, FP8, BF16} set. **Mechanism:** like IQ codebooks, weight vectors of
  8 consecutive weights are clustered into a small dictionary (tens of thousands of 8-d entries,
  shared per layer, effectively free); each group costs one index. The twist that makes it fast on
  GB10: **every dictionary entry is constrained to lie exactly on the FP8 or NVFP4 grid the matrix
  hardware natively understands**, so "decompression" is a plain table lookup that emits a standard
  FP8/NVFP4 tensor — the multiply then runs at **full tensor-core speed** (Marlin path), not a
  general-purpose gather. Reported overhead: **~10% decode, ~30% prefill**. Targets the gap below
  NVFP4 (4.5 bit floor) for 300B-class models on a single Spark. Single source; no independent
  benchmark yet. Queued for hardware verification (see roadmap).
- **[conjecture]** **Qwen3.6-27B PrismaQuant-GridBook 5.5-bit** (S-forum-gridbook): released HF
  checkpoint `rdtand/Qwen3.6-27B-prismaquant-codebook-5.5bit-vllm`. Claims **KL 0.0049 — 77% lower
  than PrismaAura at the same 5.5-bit rate** (AURA was itself strong). Uses K12–K24 (NVFP4) and
  K28–K48 (FP8) codebooks + plain NVFP4/FP8/BF16; no signed-bit formats yet. ToolEvalBench "not
  quite as high as AURA, still super strong." ~15% smaller than the prior 5.5-bit release.
- **[conjecture]** **Hy3-295B-A21B PrismaQuant-GridBook 2.9-bit** (S-forum-gridbook): released HF
  checkpoint `rdtand/Hy3-295B-A21B-prismaquant-codebook-2.9bit-vllm` — a 295B MoE on a single Spark
  via sub-NVFP4 codebook rates. OP notes it doesn't yet leverage all 41 formats; an improved version
  expected "within a few days." No tok/s reported. Corroborates the directional thesis that
  sub-4.5-bit codebook formats unlock single-Spark serving of 300B-class MoE (see also
  `[[wiki/models/laguna-s-2.1.md]]` retirement note: output quality below MiMo/DeepSeek at low
  rates — quality-vs-density tradeoff is the open question for codebook quants).
- **[conjecture]** **MTP-head quant optimizer** (S-forum-gridbook): a new PrismaQuant component that
  picks the optimal format for the MTP head specifically (vs. quantizing it like the rest of the
  model). BF16 MTP heads are most accurate but heavy; the optimizer can select any of the 41 gridbook
  formats to balance speed and accuracy of the speculative-decoding driver. Relevant to the
  documented MTP-needs-cudagraphs and MTP-OOM findings (`[[wiki/cudagraphs-and-compile.md]]`).
- **[conjecture]** **GGUF IQ formats on vLLM found lacking on GB10** (S-forum-gridbook, tenari):
  tried GGUF IQ formats inside PrismaQuant/vLLM; prefill is "just terrible" because the formats are
  platform-agnostic with no native-performance path. Consistent with the existing finding that GGUF
  is a llama.cpp path, not a vLLM fast path on GB10 (`[[wiki/llama-cpp-rpc.md]]`). MXFP8 also
  abandoned from PrismaQuant's menu ("just awful unless you're doing initial training") — replaced
  with FP8 + NVFP4 + BF16 + gridbook codebooks.
- **[conjecture]** **REAP expert pruning ineffective on modern MoE** (S-forum-gridbook, tenari):
  attempted REAP pruning but "the experts coming out of the labs are SO optimized that it's hard to
  find redundant weights to prune." Abandoned. Relevant to the documented 15% expert-prune finding
  for GLM-5.2 (S-forum-glm52-4x) — that worked on a specific model; the general claim that pruning
  is a reliable lever is weakened.
