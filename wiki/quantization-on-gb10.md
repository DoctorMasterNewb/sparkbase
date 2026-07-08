# Quantization on GB10

> **area:** quantization
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-m3-vision, S-nemotron-rpc, S-diffusiongemma
> **updated:** 2026-07-08

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
