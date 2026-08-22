# Holo-3.1-35B-A3B (computer-use VLM)

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun5, S-sm121-nvfp4
> **updated:** 2026-08-22

`Hcompany/Holo-3.1-35B-A3B` — computer-use vision-language model (Qwen3.5 VL MoE, 3B active/token,
hybrid linear+full attention, full-attention every 4th layer). Deployed behind a live computer-use
endpoint. A clean worked example of "NVFP4 MoE is the GB10 sweet spot."

## Working config (the winner: NVFP4)

- Checkpoint `Hcompany/Holo-3.1-35B-A3B-NVFP4` (ModelOpt, FP8 KV scheme, 3 shards, 23 GB disk / 20.4
  GiB resident).
- Image **`vllm/vllm-openai:gemma4-unified`** (the omni image's old loader `KeyError`s on the NVFP4
  per-expert scales — see `[[wiki/containers-and-tooling.md]]`).
- Serve: `--dtype bfloat16 --gpu-memory-utilization 0.80 --max-model-len 32768 --trust-remote-code`.
  **`--dtype bfloat16` mandatory** (config declares float32). Attention FLASHINFER; Marlin FP4 MoE.
- **[proven]** **~77 tok/s** decode single-stream; aggregate saturates ~**899 tok/s @ 128
  concurrency** (compute-bound Marlin FP4 decompress; knee ~64). Cold start ~4 min (NVFP4 weight load
  ~108 s). ⚠ **2026-08-22:** the saturation is the *Marlin dispatch*, not the silicon — sm_121 has
  native block-scaled FP4 MMA, so this ceiling is retestable here rather than inherent.
  (S-sm121-nvfp4, `[[wiki/quantization-on-gb10.md]]`)

**[proven]** The **FP8** variant (`-A3B-FP8`, compressed-tensors block-scale, 35.6 GiB) only does ~38
tok/s (Marlin FP8 fallback — `[[wiki/quantization-on-gb10.md]]`); use only if the gemma4-unified image
is unavailable.

## Computer-use integration facts

- **[proven]** Outputs coordinates in **0–1000 normalized** space (Qwen-VL convention → multiply by
  W/1000, H/1000).
- **[proven]** Chat template is in a **separate `chat_template.jinja`** (tokenizer_config is empty).
  Wraps answers in `<think>…</think>` by default.
- **[proven]** Grounding is accurate + discriminating (~87.5% on diverse synthetic UIs; separated
  adjacent buttons within ~10 px).

## Highest-leverage optimization: turn thinking OFF for grounding

The ~190-token `<think>` block is pure overhead for coordinate lookup. `chat_template_kwargs:
{"enable_thinking": false}` (injects a closed `<think></think>`).

- **[proven]** → **7.2× lower per-step latency, 4.2× throughput, 13.5× fewer output tokens, no
  grounding loss** (8/8 vs 7/8). Run grounding/execution no-think; reserve thinking for genuine
  planning turns (keep it per-request, not a server default).
- **[proven]** **Vision concurrency** (distinct 1280×800 screenshot/req, grounding action) is
  **prefill-bound**: peaks ~1.69 steps/s @ 32; interactive sweet spot **4–8 concurrent agents** (TTFT
  p95 ~8 s at 32).
- **[proven]** Download Holo with `HF_HUB_DISABLE_XET=1` (Xet finalization hangs —
  `[[wiki/containers-and-tooling.md]]`).

## Open
Native tool/function-calling smoke test not yet run; combine no-think + screenshot downscaling to push
vision concurrency further (residual bottleneck is image prefill). See `[[wiki/roadmap.md]]`.

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/containers-and-tooling.md]]` · `[[wiki/benchmarks.md]]`
