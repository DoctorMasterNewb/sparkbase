# MiMo-V2.5-NVFP4

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-mimo-results, S-mimo-doc, S-dflash-nvfp4, S-forum-mimo-2x, S-forum-mimo-3x, S-forum-mimo-tp2-1m, S-forum-mimo-dflash-22-67, S-forum-mimo-dflash-v024
> **updated:** 2026-07-09

MiMo-V2.5 — 310B Omni MoE, **MIXED_PRECISION = MXFP8 dense + NVFP4 experts**, DiffKV
(`v_head_dim=128 ≠ head_dim=192`), text+vision. The production cluster MoE. Base
`lukealonso/MiMo-V2.5-NVFP4` is the **validated, coherent** checkpoint.

## Working config

- **[proven]** Live deployment is the 1M-ctx NVFP4-KV Ray recipe; a no-ray fp8-KV recipe is the
  proven-coherent baseline.
- Image: **`vllm-node-mimo-dev39`** (vLLM 0.22.1rc1.dev305, PR#41797 merged). Do **not** swap for the
  rebuilt dev309 — the mods are verified against dev39.
- **[proven]** Attention: **`TRITON_ATTN_DIFFKV`** (DiffKV). KV `fp8`. MoE auto-selects
  **FLASHINFER_CUTLASS**; MXFP8 dense uses FlashInferCutlassMxfp8. Multinode: no-ray TP=2 (Ray hangs —
  except the ray-1m recipe which runs Ray OK). Single-node MoE **cudagraph capture is fine** (no-ray).
- Mods (all required): `fix-mimo-config`, `fix-mimo-v2-vllm`, `fix-mimo-mxfp8-dispatch`,
  `fix-mimo-qkv-mxfp8`, `fix-mimo-vision-merger`, `drop-caches`.
- **[proven]** Served as `MiMo-V2.5-NVFP4`. Full load 37/37 shards, ~86 GiB, ~19 min.

## Why the mods (the fix chain — generic GB10 lessons)

These recur on other NVFP4/MXFP8 MoE bring-ups (`[[wiki/quantization-on-gb10.md]]`):
1. **[proven]** **QKV blind-chunk corruption** — checkpoint stores fused QKV as `[Q_all|K_all|V_all]`;
   blind `chunk(tp_size)` corrupts K/V rows → multilingual garbage. Fix:
   `QKVParallelLinear.weight_loader`.
2. **[proven]** **`packed_modules_mapping` missing on the Omni wrapper** → fused `gate_up_proj` resolves
   to Unquantized, MXFP8 bytes read as bf16 → gross garbage. This fix took output garbage → coherent.
3. **[proven]** **MXFP8 not dispatched** — build dispatches FP8/NVFP4/W4A16 but not MXFP8 → MXFP8 layers
   fall to Unquantized. Add MXFP8 to `ModelOptMixedPrecisionConfig.get_quant_method`. Don't invert
   `weight_scale_inv` (UE8M0 scale metadata).
4. **[proven]** Config registry (`MimoV2Config` — transformers doesn't know `mimo_v2`); vision merger
   `bias=True`.
5. **[proven]** **SemLock spawn race** (original mod's `pip install librosa` → joblib/loky semaphores
   don't survive vLLM spawn → SemLock FileNotFoundError → rendezvous hang). Drop the audio install; clean
   stale `/loky-*` and `/dev/shm` semaphores between runs.

## Abliteration verdict (a reusable diagnostic)

**[proven]** The abliterated `lovesenko/mimo-v25-nvfp4-abliterated` **degenerates** (correct for ~15
tokens, then repetition / foreign-language) — across **two vLLM versions (0.22.1 + 0.23.1) and two mod
implementations**, while the **base** `lukealonso` checkpoint on the *same harness* is fully coherent
(including vision). ⟹ **the abliteration damaged the checkpoint; the serving stack is correct.**
Lesson: when output degenerates over length but short answers are right and it persists on raw
`/completions` (not just chat), suspect the **checkpoint**, not the template/sampling — and isolate by
serving the base model on the identical harness.

## Startup markers (sanity)

`Resolved architecture: MiMoV2OmniForCausalLM`; `Using FlashInferCutlassMxfp8LinearKernel`;
`Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend`; `Using TRITON_ATTN_DIFFKV`; `fp8_e4m3 … kv cache`.

## DFlash speculative decoding + NVFP4 KV + 1M context (custom_class, S-dflash-nvfp4)

**[proven]** **Verified 2026-07-06: DFlash spec-decode runs on the NVFP4 target with a full 1M-token
pool.** accept ~2-3 tok/block short ctx (**num_spec=5** optimal, not block_size-1=7: math 38.5 / json
42.2 / code 29.9 tok/s vs ~10 no-spec), **0.86 tok/block @ 100k depth** (prose; 8.8 tok/s), coherent +
correct needle recall to 89k.

- **[proven]** **The wall:** the native DFlashProposer registers the drafter's KV layers into the global
  paged allocator; the bf16 drafter page can't unify with the 4-bit nvfp4 target page (non-integer ratio,
  no strided-padded-read backend on GB10) -> `NotImplementedError` in `unify_kv_cache_spec_page_size`.
  See `[[wiki/attention-and-kv-cache.md]]` (the general spec-decode + quantized-KV rule).
- **[proven]** **The fix:** run the drafter as a vLLM **`custom_class`** proposer (`--speculative-config
  method=custom_class,model=dflash_custom_proposer.DFlashCustomProposer`). It is created WITHOUT the
  draft-model-runner registration -> its KV never enters the global spec -> the nvfp4 target keeps
  its full pool (spike: **1.2M-token pool**, stub drafter). Mod `dflash-custom-proposer`.
- **[proven]** **Drafter conditioning (the reference `dflash/dflash.py` is INCOMPLETE — match
  `qwen3_dflash.py`):**
  1. **Partial rotary** — rotary_dim = head_dim*partial_rotary_factor = **64**, theta =
     `dflash_config.backbone_rotary_base` = **5e6** (reference wrongly applies full-dim rotary @10000).
     This alone took accept 0.01 -> 0.44.
  2. **Dense context at REAL positions** — buffer the target aux-hidden at every committed position
     (incl. the prompt) keyed by `common_attn_metadata.positions`, committed-only. The naive `C-k`
     position inference mis-places hiddens during spec-verify steps (scheduled tokens there are the
     *speculative* block). Fixing this took accept 0.35 -> **2.08**. (Biggest single fix.)
  3. attention **sinks** (per-head bias, trained, ignored by the reference), **value_scale** 0.612,
     **mask_embedding.pt** sidecar (the `<|MASK|>` vocab row is near-zero).
  5. **Draft slice = FIRST num_spec block positions** `[1:1+num_spec]` (block[0]=seed);
     slicing from the end only works at num_spec=block_size-1. num_spec=5 > 7 by +15-20%.
- **[proven]** **Aux capture:** the target must emit aux hidden states -> `SupportsEagle3` on
  `MiMoV2Flash`/`Omni` classes + 3 `gpu_model_runner` core edits (enable aux for custom_class; bind runner
  in `_setup_eagle3_aux_hidden_state_outputs`; pass full aux + `query_start_loc` + `positions` to
  `propose()`). All in the mod's `run.sh`.
- **[proven]** **Full 1M needs GMU 0.88** (drafter is 2.9 GB loaded per rank): 0.84 fits only ~700k;
  **0.88 -> 2,106,813-token pool = 1M single-stream @ 2.11x concurrency.** Image `vllm-node-mimo-dev39`
  (nvfp4 target via `nvfp4-kv-diffkv` mod, block-size 32). Drafter staged on both nodes under the HF
  cache; download only `dflash/*` (~2.9 GB), NOT the 328 GB fp8 backbone.
- **[proven]** **NVFP4 KV is FINE at depth** — the "degradation" was a *testing artifact*:
  degenerate/repetitive filler prompts make the model emit immediate-EOS (legitimate behavior on
  pathological input), NOT nvfp4 corruption. With real varied content it is coherent to 89k+ (the
  no-DFlash nvfp4 target behaves identically). Lesson: test deep context with realistic varied text (this
  is why `llama-benchy` uses a Gutenberg book). ⟹ the prior 1M MTP+nvfp4 recipe does **not** fall apart at
  depth either — same KV path.
- **[conjecture]** **fp8 KV alternative** (forum HeNryous, `mimo-v25-dflash-dgx-spark`): target fp8 + bf16
  draft + native dflash on newer vLLM (dev760); fp8's 8-bit page unifies more cleanly than nvfp4's 4-bit,
  so no custom_class needed there. Not required for us — nvfp4 meets the goal. fp8 kernel is a trivial
  in-register `bitcast uint8->fp8->bf16 * descale` if we ever port it.

## vLLM v0.24.0 port (2026-07-07, S-dflash-nvfp4)

**[proven]** **DFlash + NVFP4 KV works on v0.24.0 (0.24.1.dev0)** — accept 2.37/block, nvfp4 KV pool
1.35M tok, needle@26k, coherent.

What v0.24.0 UPSTREAMS (dev39 mods that DROP): MiMoV2Omni model + config arch registry (registry.py),
DiffKV backend (#41797, both triton + new flash_attn_diffkv), MXFP8 kernels (modelopt.py), native DFlash
+ `custom_class_proposer.py`.

What STILL needs mods for the lukealonso NVFP4 export (its MXFP8 fused-qkv layout ≠ Xiaomi official):
- **[proven]** `fix-mimo-config` — register MimoV2Config (transformers still doesn't know mimo_v2).
- **[proven]** `fix-mimo-qkv-mxfp8` — dev39 mod APPLIES UNCHANGED: QKV-aware loader splitting fused
  [Q_all|K_all|V_all] via param.weight_loader + `weight_scale_inv` alias + MXFP8 dispatch in
  MIXED_PRECISION get_quant_method. (v0.24.0's native `_shard_fp8_qkv_proj` assumes fp8 block 128; this
  checkpoint is MXFP8 block 32 → garbage without this.)
- **[proven]** `fix-mimo-v0240-merger` — MiMoVisionPatchMerger mlp.0/mlp.2 bias=True (KeyError else).
- **[proven]** `fix-mimo-v0240-packed` — packed_modules_mapping on the Omni WRAPPER (else fused
  qkv/gate_up → Unquantized → MXFP8 as bf16 → multilingual garbage).
- **[proven]** `nvfp4-kv-diffkv` — backend + attention.py patch drop in; the KERNEL needed a 1-line fix:
  v0.24.0's `compute_kv_seq_mask` (triton_attention_helpers.py) added a `seq_len` positional arg the dev39
  kernel call omitted.
- **[proven]** `dflash-custom-proposer` — APPLIES UNCHANGED (runner anchors a/b/c + mimo_v2_aux.diff +
  omni SupportsEagle3 all match). Native DFlash still can't do nvfp4 KV (registers draft KV → page-unification
  wall; MiMo lacks native SupportsEagle3), so custom_class remains the path even on v0.24.0.

Lesson: on a version bump, most MiMo mods either upstream or apply unchanged; the churn is
checkpoint-specific quant loaders (qkv/merger/packed) + triton kernel arg drift.

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/attention-and-kv-cache.md]]` · `[[wiki/multinode-tp-and-networking.md]]`

## Forum ingest: community recipes & TP=3 virtual-head padding (2026-07-08)

- **[reported]** **TP=3 across 3× DGX Spark** (S-forum-mimo-3x, tonyd615): MiMo has 64 attention
  heads / 4 KV heads — neither divides by 3, so stock vLLM can't TP-shard across 3 nodes. Fix:
  **virtual-head padding** — pad to 96 query / 6 KV heads (32 q / 2 KV per rank), zero-mask the pad
  heads so they contribute nothing. Same approach used for MiniMax-M3 TP=3. Two additional fixes:
  FusedMoE zero-fill (uninitialized padded MoE tail corrupted NVFP4 output) and attention_sink_bias
  padding fix for the MTP draft (loader did 64//3=21 while virtual sink pads to 32).
  - Results (thinking OFF, 3-run avg): quality 97.3, decode **38.8 tok/s** (effective 35.1), KV
    cache 3,127,938 tokens at 1M context, all 4 modalities verified live.
  - **[reported]** Thinking ON vs OFF: OFF wins (97.3 vs 88.9 quality, 2× lower answer latency).
    Thinking ON only posts higher raw tok/s because it generates internal reasoning tokens.
  - Infra: Ray executor with `object-store-memory capped to 1GB + memory-monitor disabled` (GB10
    unified memory sits near full when loaded, which is normal), worker-first launch, MTU 9000.
- **[reported]** **TP=2 with NVFP4 KV + 1M context** (S-forum-mimo-tp2-1m, tonyd2wild): ~30 tok/s,
  NVFP4 4-bit KV (~1.01M-token pool). 69-eval: thinking-OFF 97.8 beats thinking-ON 90.6 for
  tool/agent work. Quality did not degrade with NVFP4 quant.
- **[reported]** **TP=2 community recipe** (S-forum-mimo-2x, a3refaat): vLLM 0.21.1rc1.dev39,
  `--distributed-executor-backend ray`, `--load-format instanttensor`,
  `--attention-backend triton_attn_diffkv`, `--kv-cache-dtype fp8_e4m3`, 131072 ctx, MTP
  `num_speculative_tokens=2`. Reported benchmarks: Q&A 36.9, code 39.9, JSON 41.9, math 33.5 tok/s
  (run 2). Image input validated. Uses eugr/spark-vllm-docker PR #251.

## Forum ingest: DFlash 22→67 tok/s, v0.24.0 DFlash+NVFP4 KV (2026-07-09)

- **[reported]** **DFlash spec-decode acceptance scales with output structure** (S-forum-mimo-dflash-22-67,
  danielgbates): on 2× Spark TP=2 eager, DFlash gives **22.3 tok/s no-spec → 27.6 prose / 45.4 code /
  55.1 math / 66.9 JSON** with DFlash. Acceptance scales with output structure (JSON > math > code >
  prose), explaining the "low acceptance anomaly" reported elsewhere. The drafter is worth it on Spark
  specifically because cross-node TP=2 pays ~48 cross-node all-reduces per eager decode step — a
  drafter getting 3-6 accepted tokens/step amortizes the fixed per-step latency. DFlash drafts a block
  of 8 masked tokens in one forward (vs EAGLE's sequential token-by-token). Drafter: 5-layer qwen3-arch,
  hidden 4096, SWA-1024, block size 8, cross-attends to backbone layers [0,11,23,35,47], 2.9 GB.
  Target: `lukealonso/MiMo-V2.5-NVFP4` (~170 GB). Requires vLLM ≥ 0.23.1 nightly with PRs #45200,
  #45181 (mixed KV page sizes), #46104 (SWA + DFlash for MiMo). GitHub: `DoctorMasterNewb/vLLM-Mimo-V2.5-Dflash-2x-DGX-Spark`.
- **[reported]** **DFlash + NVFP4 KV in one vLLM v0.24.0 instance** (S-forum-mimo-dflash-v024,
  danielgbates): the `custom_class` proposer approach (already first-party proven in S-dflash-nvfp4)
  is confirmed working on v0.24.0 upstream — the drafter's KV never enters the global paged allocator,
  so NVFP4 target KV + DFlash drafter coexist. v0.24.0 upstreams most old MiMo mods (MiMoV2Omni,
  DiffKV #41797, MXFP8 kernels, native DFlash/custom_class). Still needs startup mods for the
  lukealonso NVFP4 export: `fix-mimo-config`, `fix-mimo-qkv-mxfp8`, `fix-mimo-v0240-merger`,
  `fix-mimo-v0240-packed`, `nvfp4-kv-diffkv` (1-line triton kernel fix for `seq_len` arg),
  `dflash-custom-proposer`. GitHub: `DoctorMasterNewb/vLLM-MiMo-V2.5-DFlash-NVFP4Kv`.
