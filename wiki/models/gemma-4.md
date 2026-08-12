# Gemma-4

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-forum-gemma4-26b-bench, S-forum-spark-field-notes
> **updated:** 2026-08-12

Gemma-4 family on GB10. The 12B "unified" variant is a clean case study in *image-support* deciding
serveability, and in FP8 online-dynamic being the GB10 fast path for a dense model.

## gemma-4-12B-it (the new unified arch)

- **[proven]** Arch **`Gemma4UnifiedForConditionalGeneration` / `model_type: gemma4_unified`** —
  encoder-free omni (text+vision+**audio** towers, text hidden_size 3840, 48 layers). bf16
  single-file ~24 GB.
- **[proven]** **Too new for older images:** transformers < 5.10 and vLLM < the d20260603 build don't
  register the *unified* arch; Atlas has no AOT kernel for hidden 3840. **Use
  `vllm/vllm-openai:gemma4-unified`** (support landed transformers v5.10.0 + vLLM PR #44429).
- Serve text-only fast: `--gpu-memory-utilization 0.55 --max-model-len 32768 --limit-mm-per-prompt
  '{"image":0,"audio":0}'` (skips MM profiling).
- **[proven]** **FP8 online dynamic (`--quantization fp8`) is the recommended mode** — native CUTLASS
  path, **2.08×** bf16 decode (7.6 → 15.8 tok/s), 22.3 → 12.9 GiB, no quality loss in spot-checks.
  This is the GB10 FP8 fast path (NOT a block-scale checkpoint). See
  `[[wiki/quantization-on-gb10.md]]`.
- **[proven]** Quality note: strong reasoning/format/code; **weak factual calibration** — states
  partial facts confidently (model trait, present in both bf16 and FP8, not a quant artifact).

## gemma-4 NVFP4 variants (via Atlas)

- **[proven]** `nvidia/...gemma-4-31b-nvfp4` — dense (sliding+full attn, GeGLU), fp8 KV,
  `tool_call_parser: gemma4`, **~9 tok/s** (dense = slow). Creative gen drifts into repetition at
  greedy → clients should send `min_p: 0.05` + `repetition_penalty: 1.1`. Long ctx to 16K clean.
- **[proven]** `bg-digitalservices/gemma-4-26b-a4b-nvfp4` — hybrid attn + MoE, GeGLU, fp8 KV,
  **~67 tok/s** (MoE wins again — `[[wiki/quantization-on-gb10.md]]`).
- Atlas actively maintains Gemma-4 kernels (dispatch + v_norm/lm_head/o_proj fixes through 2026-05).

## gemma-4-26B-A4B NVFP4: Unsloth vs nvidia (vLLM, forum benchmark)

- **[conjecture]** **Unsloth NVFP4 ~17% faster than nvidia NVFP4 on DGX Spark** (S-forum-gemma4-26b-bench,
  shahizat): benchmarked `unsloth/gemma-4-26B-A4B-it-NVFP4` vs `nvidia/Gemma-4-26B-A4B-NVFP4`
  with identical vLLM serve params (`--kv-cache-dtype fp8 --max-model-len 65536 --max-num-seqs 8
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.85 --reasoning-parser gemma4
  --enable-prefix-caching --trust-remote-code`), `vllm bench serve` with 100 concurrent random
  1000-in/1000-out requests. On DGX Spark: Unsloth **159.77 tok/s aggregate output** (mean TPOT
  47.14 ms) vs nvidia **128.21 tok/s** (mean TPOT 58.67 ms) — Unsloth ~17% higher throughput,
  ~15% lower TPOT. This corroborates the broader pattern that Unsloth NVFP4 quants behave
  differently from nvidia NVFP4 on GB10 (cf. S-forum-unsloth-qwen36 where Unsloth was ~15%
  *slower* for Qwen3.6-35B-A3B — the direction is model-specific, not uniformly better).
  - **[conjecture]** **DGX Spark ~6-7× slower than RTX Blackwell 6000 Pro** on this workload:
    Blackwell 6000 Pro hits 901-1058 tok/s aggregate output vs Spark's 128-160 tok/s. The 6000
    Pro also has ~5× lower TPOT (7-8 ms vs 47-59 ms). This is consistent with the known
    bandwidth-bound decode ceiling on GB10 (~270 GB/s vs the 6000 Pro's much higher bandwidth).
  - **[conjecture]** These are aggregate throughput numbers at 100 concurrent requests, not
    single-stream decode. The TPOT (~47-59 ms on Spark) implies ~17-21 tok/s per-stream, which
    is lower than the Atlas ~67 tok/s above — likely due to the concurrency saturating the
    bandwidth-bound decode path. See `[[wiki/benchmarks.md]]`.

## gemma-4-26B-A4B NVFP4: cross-engine field-notes benchmark (2026-08-12 ingest)

- **[conjecture]** **nvidia NVFP4 30.3 tok/s single-stream (no MTP) vs community Q4_K_M GGUF
  49.6 tok/s in Ollama — "the obvious A/B makes the NVFP4 checkpoint look slower than it is"**
  (S-forum-spark-field-notes, ss121): measured on a single DGX Spark (vLLM 0.26.1rc1.dev535,
  driver 580.173.02, CUDA 13.0) with an identical harness across Ollama and vLLM. With MTP
  enabled, the NVFP4 checkpoint moves to 54.9 tok/s (+81%). The OP's point: a naive comparison
  (NVFP4 without MTP vs GGUF Q4_K_M) makes the official NVIDIA NVFP4 checkpoint appear slower
  than the community GGUF — but MTP closes and exceeds the gap. This is consistent with the
  proven finding that speculative decoding pays off unusually well on bandwidth-bound hardware
  (verification is nearly free when weights had to be read anyway).
  - **MTP `num_speculative_tokens: 2` beats 4** — per-position acceptance decays
    0.84 → 0.60 → 0.39 → 0.27, so positions 3-4 add cost without proportional benefit. This
    corroborates the existing [proven] finding on DSV4-Flash-DSpark (nst=3 beats 5) and the
    GLM-5.2 finding (MTP4 beats MTP5 on tool-eval-bench).
  - vLLM at 8 concurrent: 303.6 tok/s aggregate for Gemma-4-26B-A4B+MTP (vs Ollama 51.9 — Ollama
    does not batch, so its 8× aggregate equals its single-stream figure). TTFT: vLLM 94 ms vs
    Ollama 511 ms.

## Open
Whether Atlas's gemma4 loader accepts the **unified** (audio-tower) arch vs the text+vision
`Gemma4ForConditionalGeneration` of the stock recipes — bring-up was cut off, unresolved
(`[[wiki/roadmap.md]]`).

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/engines.md]]` · `[[wiki/benchmarks.md]]`
