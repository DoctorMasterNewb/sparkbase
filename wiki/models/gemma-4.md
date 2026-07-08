# Gemma-4

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4
> **updated:** 2026-07-08

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

## Open
Whether Atlas's gemma4 loader accepts the **unified** (audio-tower) arch vs the text+vision
`Gemma4ForConditionalGeneration` of the stock recipes — bring-up was cut off, unresolved
(`[[wiki/roadmap.md]]`).

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/engines.md]]` · `[[wiki/benchmarks.md]]`
