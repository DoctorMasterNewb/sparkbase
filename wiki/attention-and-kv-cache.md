# Attention backends & KV cache on GB10

> **area:** attention
> **status:** stable
> **evidence:** proven
> **sources:** S-m3-vision, S-mimo-results, S-mimo-doc, S-sess-jun5, S-sess-jun4, S-dflash-nvfp4, S-forum-mimo-2x-opt, S-forum-dsv4-kvcache, S-forum-inkling-nvfp4
> **updated:** 2026-07-20

Which `--attention-backend` to pass is decided by the model's attention type, not preference. Get it
wrong and KV-cache init fails or numerics are subtly off.

## Backend selection

| Backend | Use when | Notes on GB10 |
|---|---|---|
| **TRITON_ATTN** | sparse / MSA / block_size 128 archs (MiniMax-M3 MSA) | **required**, not optional, for MSA — supports `block_size=128` and is numerically correct on Blackwell. |
| **TRITON_ATTN_DIFFKV** | DiffKV archs where `v_head_dim ≠ head_dim` (MiMo-V2.5: v=128, head=192) | the MiMo path; from vLLM PR #41797, native in dev39-era images. |
| **FLASHINFER** | standard dense/MoE attention (Qwen3.x, Holo, Gemma-4) | the common fast path; default for ModelOpt NVFP4. |
| FlashAttention (ViT) | vision towers | first image triggers a one-time JIT autotune (~20 s) — see below. |

## Hard rules

- **[proven]** **`--block-size 128` + `TRITON_ATTN` for MSA / sparse-attention models.** FLASHINFER for
  the dense layers has **no common block size** with the MSA sparse/index cache → KV init fails ("No
  common block size for 128"). TRITON_ATTN supports 128. (MiniMax-M3.)
- **[proven]** **DiffKV needs TRITON_ATTN_DIFFKV.** `v_head_dim ≠ head_dim` checkpoints (MiMo-V2.5) won't
  run on a vanilla backend.
- **[proven]** **KV-cache dtype `fp8` (e4m3) is the default and is precise enough.** bf16 KV is *more*
  precise but rarely needed; on GB10 KV is cheap for sparse/hybrid archs anyway (few KV heads). MiMo uses
  `--kv-cache-dtype fp8`; M3's MSA + 4 KV heads make even 40k+ ctx cheap.
- **[proven]** **Sparse/hybrid attention makes long context cheap.** MSA (M3, 4 KV heads), Mamba-2
  hybrids (Nemotron-3: most layers SSM, few attn layers w/ 2 KV heads) → huge context for little memory
  (Nemotron-3 ran the model's full 1M ctx × 4 slots). The KV pool is rarely the constraint; **weights
  are**.

## Gotchas

- **[proven]** **First image request = ~20 s ViT JIT autotune** (`AttentionBackendEnum.FLASH_ATTN for vit
  attention`). Short client timeouts make the first call look like it "returned nothing." Use a long
  timeout (>150 s) or send a warmup image; every request after is fast. (M3 vision, Holo vision.)
- **[proven]** **flashinfer gemma-rmsnorm CUTLASS-DSL ICE.** Archs with `use_gemma_norm=true` call
  `flashinfer.norm.gemma_rmsnorm`, whose CUTLASS-DSL kernel can fail MLIR verification on cu130
  (`'llvm.mlir.global_dtors' requires attribute 'data'`). Fix: swap in pure-torch norms (verified
  coherent). Was misdiagnosed as a vision bug — it's a text-path norm. (MiniMax-M3.)
- **[proven]** **`--dtype bfloat16` may be mandatory** even when the checkpoint config says `float32`
  (Holo FP8/NVFP4 configs declare float32 — force bf16 activations or the launch is wrong).
- **[proven]** **Atlas KV flags differ** from vLLM (no `--kv-cache-gb`): `--max-seq-len`, `--block-size`,
  `--kv-cache-dtype`, `--kv-high-precision-layers`, `--ssm-cache-slots`, etc. See `[[wiki/engines.md]]`.

- **[proven]** **Spec-decode drafter + QUANTIZED target KV = page-unification wall.** A native spec
  proposer (DFlash/EAGLE3/MTP draft-model) registers the drafter's KV layers into the *global* paged
  allocator, which requires ONE physical page size across all layers. A bf16 drafter page can't unify with
  a 4-bit **nvfp4** target page (non-integer ratio; and NO GB10 backend implements the strided-padded-read
  `indexes_kv_by_block_stride` path — it's `False` on every backend) ->
  `NotImplementedError: The page size of the layer is not divisible by the maximum page size` in
  `unify_kv_cache_spec_page_size`. **Workaround:** run the drafter as a **`custom_class`** proposer so its
  KV is never registered globally (drafter runs standalone) -> the quantized target keeps its full pool.
  `fp8` (8-bit) target unifies more cleanly (2x bf16 ratio) so the native path can work there. Status:
  `workaround` — see `[[wiki/models/mimo-v2.5.md]]` (DFlash). Reusable for any spec-decode + quantized-KV
  combo on GB10.
- **[proven]** **Deep-context coherence tests need REALISTIC text.** Degenerate/repetitive filler prompts
  make models emit immediate-EOS at depth (a real behavior on pathological input) — this looks exactly like
  KV-quant corruption but is NOT. nvfp4 KV is coherent to 89k+ with varied content. Always test depth with a
  real corpus (`llama-benchy` uses a Gutenberg book for this).

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/platform-gb10.md]]` · `[[wiki/models/minimax.md]]` · `[[wiki/models/mimo-v2.5.md]]`

## Forum ingest: TRITON_ATTN_DIFFKV quantized KV guard (2026-07-11)

- **[conjecture]** **`TRITON_ATTN_DIFFKV` raises `NotImplementedError` on quantized
  `kv_cache_dtype`** (S-forum-mimo-2x-opt, renek): the backend's guard rejects any quantized KV
  cache dtype, but the underlying store kernel already accepts dtype + scales. The guard is
  defensive — disabling it allows `--kv-cache-dtype fp8_e4m3`, which roughly doubles the KV pool.
  This conflicts with first-party findings where fp8 KV works fine with TRITON_ATTN_DIFFKV
  (proven on dev39+), suggesting the guard was relaxed in later vLLM versions. If you hit this
  error on an older build, patching the guard is the workaround.

## Forum ingest: DSV4-Flash KV cache sizing on Spark (2026-07-14)

- **[conjecture]** **DSV4-Flash KV cache ~15 GB/1M tokens/node on 2× Spark** (S-forum-dsv4-kvcache,
  paxren2020): user running DeepSeek on a dual-node cluster reports ~15 GB VRAM per 1M context
  tokens per node (~30 GB total across 2 nodes, ~15 GB allocated on each). vLLM logs show
  `Available KV cache memory: 26.15 GiB`, `GPU KV cache size: 1,687,476 tokens`, `Maximum
  concurrency for 500,000 tokens per request: 3.37x` at `--gpu-memory-utilization 0.90`. This is
  significantly larger than online KV cache calculators (kvcache.ai) and Reddit formulas predict
  (~5× smaller). Possible explanations: vLLM overhead/fragmentation, CUDA graph memory
  profiling reservation, or the calculators modeling a different attention variant. (S-forum-dsv4-kvcache)
- **[conjecture]** **CUDA graph memory profiling overhead** (S-forum-dsv4-kvcache): vLLM v0.21.0+
  enables CUDA graph memory profiling by default. The log message shows that
  `--gpu-memory-utilization=0.9000` is equivalent to `0.8943` without profiling — the profiling
  reserves ~0.6% of usable memory. To maintain the same effective KV cache size, increase to
  `--gpu-memory-utilization 0.9057`. Disable profiling with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. This overhead is on top of the standard
  cudagraph capture memory cost. (S-forum-dsv4-kvcache)

## Forum ingest: tml_fa4 Sm120 path has no paged-KV (2026-07-20)

- **[conjecture]** **The `tml_fa4` Sm120/Sm121 cute FlashAttention path has no paged-KV support**
  (S-forum-inkling-nvfp4, greg190): vLLM's paged cache cannot feed the `tml_fa4` Sm120 kernel. The
  Inkling 8× Spark bring-up worked around this by **re-gathering the whole KV history into contiguous
  buffers every decode step** (patched `fa4_rel_attention` to gather paged KV → contiguous per call).
  This workaround is O(context) per token → decode caps at ~24 tok/s aggregate at real context
  regardless of concurrency, producing a steep long-context decode cliff (25 tok/s @ ~100 tok →
  13.5 tok/s @ 2048, c1). **Why it bites on Spark:** until paged-KV lands natively in the Sm120/Sm121
  cute FA4 path, any model routing through `tml_fa4` on GB10 will hit this cliff. This is the
  load-bearing blocker for Inkling (and likely other rel-bias / FA4-arch models) on sm_121a. The OP
  also notes the Sm80-inherited rel-bias path **discards the relative-position bias** (no bias
  parameter) → plausible-but-wrong outputs; the **score-mod `vllm_flash_attn/cute` path is the
  intended sm12x route.** See `[[wiki/models/inkling.md]]`. Single source → [conjecture], but a
  filed upstream issue (vllm#49049) and public patch set accompany it.
