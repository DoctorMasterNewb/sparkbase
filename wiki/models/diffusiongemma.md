# DiffusionGemma-26B-A4B

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-diffusiongemma, S-swapper, S-sm121-nvfp4
> **updated:** 2026-08-22

DiffusionGemma is a **discrete / block-diffusion** language model (NOT autoregressive): 26B MoE
(~4B active), Gemma4 backbone, bidirectional attention. It generates by iteratively **denoising a
fixed-length token canvas** (256-token blocks) rather than emitting one token at a time.

- **[proven]** Served TP=2 on the GB10 pair on the standard serving port. Two checkpoints exist:
  `google/diffusiongemma-26B-A4B-it` (bf16 base) and `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`
  (a quant of it).

## Serveable natively — no mod needed
- **[proven]** A recent `vllm-node` image (vLLM `0.23.1rc1.dev520`) ships **full native support**:
  `diffusion_gemma.py`, `DiffusionGemmaForBlockDiffusion` in the model registry, `DiffusionConfig`
  in `config/diffusion.py`, and the `gemma4` parser (`vllm/parser/gemma4.py`). The eugr
  `mods/diffusiongemma` patch chain (175 KB `diffusiongemma-support.patch` + attention/parser
  patches) is **only for images that lack this** — its `run.sh` self-probes
  `has_upstream_diffusiongemma_support()` first. On an image with native support, skip the mod.

## Required serve flags (the family settings)
```
--attention-backend TRITON_ATTN          # bidirectional diffusion attention; the default won't do
--diffusion-config '{"canvas_length":256}'
--enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4
--override-generation-config '{"max_new_tokens": null}'
```
- **[proven]** The chat template embedded in the model's `tokenizer_config.json` works — do **not**
  pass `--chat-template` (that needs the mod's `fixed_chat_template.jinja`, which isn't shipped in
  the modless path).

## Issue: NVFP4 MoE crashes engine init on the default backend
- **[proven]** **Symptom:** engine core dies at init — `NotImplementedError: ('Intermediate size
  padding for w1 and w3, for %s NvFp4 backend, but this is not currently supported',
  'FLASHINFER_CUTLASS')`. The API server then exits with "Engine core initialization failed" and the
  serving port never comes up.
- **Root cause:** with no NVFP4 env set, vLLM defaults the NvFp4 MoE to `FLASHINFER_CUTLASS`, which
  cannot pad this model's `w1`/`w3` expert intermediate size.
- **Workaround:** force the Marlin NvFp4 MoE backend with **`VLLM_TEST_FORCE_FP8_MARLIN=1`** (log
  confirms `Using 'MARLIN' NvFp4 MoE backend`). NB: `VLLM_NVFP4_GEMM_BACKEND` **does not exist in
  this image** — it warns "Unknown vLLM environment variable" and no-ops. `VLLM_USE_FLASHINFER_MOE_FP4`
  is also absent here (only `..._INT4`). See `[[wiki/quantization-on-gb10.md]]`.
- **Status:** fixed (marlin). Retired anyway — see deployment below.

## Issue: empty output at small token budgets (diffusion, not a bug)
- **[proven]** **Symptom:** short requests return **empty content** (`content: None`, or literally
  `...`). A question with `max_tokens=32` → empty; `max_tokens=128` → correct ("…**Paris**."). A
  benchmark harness's coherence probe **FAILS** on this model for the same reason ("Expected 'Paris'.
  Got: …").
- **Root cause:** block diffusion denoises the whole 256-token canvas; a tiny budget doesn't converge
  to usable text.
- **Workaround:** give it room (≥128 tokens); benchmark with a **`--skip-coherence`** equivalent. The
  model is coherent — this is a diffusion-vs-short-probe mismatch, not breakage.

## bf16 vs NVFP4 (measured 2026-07-01) — and why bf16 is the deployment
- **[proven]** bf16 (`google/…-it`) serves with **`--moe-backend triton`** and no fp4 env. NVFP4
  needs the marlin force above and is ~1/3 the memory (18 GB vs 52 GB) and ~1.8× faster prefill
  (pp512 287 vs 159 tok/s, TTFT ~1.8 s vs ~3.4 s). Full numbers + the decode-metric caveat:
  `[[wiki/benchmarks.md]]`.
- **[proven]** On raw serving perf NVFP4 was the stronger option; we deploy **bf16** as the
  full-precision choice (avoids the weight-only-marlin FP4 decompress path — GB10 has no native FP4).
  Deployed via serving-supervisor recipe (bf16 stack). **The NVFP4 recipe/unit/weights were retired
  and deleted on 2026-07-01.**
- **[superseded]** ⚠ **The rationale above is invalidated (2026-08-22, S-sm121-nvfp4).** "GB10 has no
  native FP4" is false. Worse: the specific reason this model fell to Marlin was
  `Intermediate size padding for w1 and w3`, which is the **SM120 128-alignment gate** on the native
  FP4 tile shapes — an offline checkpoint-padding problem, not a missing kernel. NVFP4 was already
  ~1.8× prefill at ⅓ the memory and was retired on a false premise. Re-open NVFP4 first if this model
  is ever wanted again. `[[wiki/quantization-on-gb10.md]]`

## Finetunes
- **[proven]** **`FredyRivera-dev/diffusiongemma-26B-A4B-it-HERETIC-Uncensored`** (abliterated/"heretic"
  decensor, full bf16) — **evaluated 2026-07-01, rejected: produces poor-quality / trash output.** The
  abliteration does lift refusals (it complies with profanity/dark-humor/forceful-opinion prompts),
  and short smoke probes can look coherent, but broader eval showed degraded output quality — not
  worth deploying. Served fine mechanically (same TP=2 recipe as bf16, `--moe-backend triton`); the
  problem is the checkpoint, not the serving. Weights deleted from both nodes. Stick with the base
  `google/diffusiongemma-26B-A4B-it` bf16 deployment.

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/attention-and-kv-cache.md]]` · `[[wiki/benchmarks.md]]` ·
`[[wiki/models/gemma-4.md]]` (the Gemma4 backbone family)
