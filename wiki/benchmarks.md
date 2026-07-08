# Benchmarks (GB10 / DGX Spark)

> **area:** benchmarks
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-m3-20tps, S-nemotron-rpc, S-mimo-doc, S-minimax-sweeps, S-swapper-sweep, S-dgxspark-report, S-diffusiongemma
> **updated:** 2026-07-08

Single-stream decode unless noted. All on the 2× GB10 pair. Numbers anchor the rules on
`[[wiki/platform-gb10.md]]` (bandwidth-bound) and `[[wiki/quantization-on-gb10.md]]` (MoE-NVFP4 wins).
Append rows as new models are benched; cite the source.

- **[proven]** Every row below is a first-party measurement on a real GB10 pair **except** rows marked
  `(vendor)`, which are **[reported]** (vendor-claimed, not independently reproduced here).

| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Mem/node | Notes |
|---|---|---|---|---|---|---|---|
| gemma-4-12B-it | bf16 | vLLM | 1 | 7.6 | 32k | 22.3 GiB | bandwidth-bound baseline |
| gemma-4-12B-it | FP8 online-dynamic | vLLM | 1 | **15.8** | 32k | 12.9 GiB | 2.08× bf16, native CUTLASS, no quality loss |
| gemma-4-31b | NVFP4 | Atlas | 1 | ~9 | 16k | — | dense (vendor) — [reported] |
| gemma-4-26b-a4b | NVFP4 | Atlas | 1 | ~67 | — | — | hybrid attn+MoE (vendor) — [reported] |
| Holo-3.1-35B-A3B | FP8 block-scale | vLLM (omni img) | 1 | 38.3 | 32k | 35.6 GiB | Marlin FP8 fallback |
| Holo-3.1-35B-A3B | NVFP4 (modelopt) | vLLM (gemma4-unified) | 1 | **76.9** | 32k | 20.4 GiB | 2× FP8; aggregate 899 tok/s @128 conc |
| Qwen3.6-35B-A3B (MoE) | NVFP4 + MTP, fp8 KV | Atlas | 1 | **142.3** (peak 144) | 256k pool | ~43–64 GB | the fast regime; service ~76–96 |
| Qwen3.6-27B (dense) | FP8 + MTP, bf16 KV | Atlas | 1 | 30.2 (peak 49.7 MTP) | 8k | ~106 GB | dense = all params/token |
| MiniMax-M2.7-abliterated | AWQ 4-bit | vLLM | 2 (EP=2) | **23.9** (peak 25 MTP) | — | — | durable default; prefill pp2048 ~1100, pp8192 ~1268 tok/s |
| MiniMax-M2.7 (cyankiwi) | AWQ 4-bit | vLLM | 2 (EP=2) | ~22 (peak 23) | — | — | prefill pp2048 ~930–1120 |
| MiniMax-M2.7 | NVFP4 (FlashInfer-CUTLASS) | vLLM | 2 (EP=2) | **16.5** (peak 18) | — | — | **decode slower than AWQ**, prefill higher (pp2048 ~1320–1500). EP=2 raw tg32 ~15.3 |
| MiniMax-M3 | AutoRound-3.2bit | vLLM | 2 (TP=2) | ~5 (prefill ~74) | 40k | 88.4 GiB | eager+cross-node bound; superseded by EAGLE3 row |
| **MiniMax-M3 PRODUCTION** | AutoRound + EAGLE3, **nvfp4 KV**, inductor compile | vLLM dev537 | 2 (TP=2 EP=2) | **12.6 prose / 15.5 code (peak 19)** (prefill 308) | **262k** (pool 359k tok) | ~91 GiB | 2026-07-04 image :nvfp4kv-compile; clean reasoning field (m3_mmthink), tools, vision, multi-turn; CG replay open |
| MiniMax-M3 + EAGLE3 + compile (fp8) | AutoRound + draft, fp8 KV, inductor | vLLM dev537 | 2 | ~13 prose / 16.3 code (peak ~20) (prefill 321) | 102k | 90.7 GiB | image :compile; slightly faster, less ctx |
| MiniMax-M3 + EAGLE3 (eager) | AutoRound-3.2bit + Inferact draft, **fp8 KV** | vLLM dev537 | 2 (TP=2 EP=2) | **13.3 prose / 15.3 code (peak 20)** (prefill 310; 340 @ 68k depth) | **102k** (pool 244k tok) | 90.7 GiB | 2026-07-03; vision + tools working; nst=3, util 0.90, mnbt 4096, video-mm off; 68k needle PASS; nst=5/draft_tp=1 WORSE (9.6/12.9); reasoning-parser OFF (−30% streaming); stack `minimax-m3-eagle3` |
| **MiniMax-M3-W4A16-GPTQ (b12x)** | GPTQ W4A16 + nvfp4 KV + EAGLE3 | vLLM b12x (a3refaat) | 2 (TP=2) | **36.3 tg32 / 34.7 tg128 (peak 41)** (pp2048 **1028**) | 32k (desktop-head; 196k headless) | ~104 GiB | 2026-07-05; forum 375595 reproduced EXACTLY (35.5 — [reported] origin); **VISION WORKS** (drop --language-model-only: 32.9 tg32 w/ ViT, exact OCR, 113k pool); needs healthy GPU clocks (wedge→20) + warm triton cache both nodes |
| Nemotron-3-Super-120B-A12B | Q8_0 GGUF | llama.cpp RPC | 2 | ~10.5 | 1M | ~61 GB | 128 GB model, fits neither node alone |

## Swapper menu — full sweep (2026-06-30, post power-cycle + cross-node fixes)

Single-methodology sweep of every validated recipe on a serving supervisor / model-swapper,
back-to-back via a bench-recipe harness (short-prompt g256 single-stream = decode rate,
`/v1/completions`). Measured **after** the DGX Spark power-controller wedge was cleared
(`[[wiki/platform-gb10.md]]`) and after fixing 4 sparkrun cross-node bugs (below) — so these
**supersede earlier suppressed numbers** (e.g. minimax **43.8** now vs the 23.9 recorded while the
head was likely wedged / not truly cross-node).

- **[proven]** All med/max tok/s below are first-party GB10 measurements. The role column names which
  node role served the recipe.

| Recipe (role) | Model | Quant | Engine | Nodes | med/max tok/s | TTFT | conc4 / conc8 agg |
|---|---|---|---|---|---|---|---|
| head text | Qwen3-Coder-Next-NVFP4-GB10 | NVFP4 | vLLM | 1 (head) | **64.0** / 64.3 | 93 ms | 44 / 306 |
| qwen36-uncensored | Qwen3.6-35B-A3B-Uncensored-Aggressive | bf16 MoE + fp8-KV | vLLM TP=2 (Ray) | 2 | **50.5** / 50.7 | 84 ms | 145 / 131 |
| minimax-m2.7-ablit | MiniMax-M2.7-abliterated-heretic-AWQ | AWQ 4-bit | vLLM TP=2 (mp) | 2 | **43.8** / 43.9 | 52 ms | 172 / **257** |
| deepseek-v4-flash | DeepSeek-V4-Flash-DSpark | NVFP4 (`nvfp4_ds_mla` KV) | vLLM+DSpark self-spec-decode TP=2 | 2 | **43.7** / 54.3 | 188 ms | 79 / 88 |
| mimo-v25 | MiMo-V2.5-NVFP4 | NVFP4 + NVFP4-KV | vLLM TP=2 (Ray) | 2 | **32.3** / 32.6 | 203 ms | 68 / 94 |
| worker omni | Nemotron-3-Nano-Omni-AEON | BF16 | vLLM (omni) | 1 (worker) | **27.3** / 27.4 | 185 ms | 114 / 114 |

Reads: single-node head-text is fastest (no host-staged cross-node all-reduce). DeepSeek's max 54.3
approaches the **[reported]** forum number ~56.7; its flat conc scaling is `max_num_seqs=6` (1M-ctx
profile). minimax scales best under load (AWQ + EP). deepseek/mimo have higher TTFT (spec-decode draft
/ 1M-ctx setup). The Nemotron omni is BF16 61.6 GiB and takes ~20 min to start (8 min load + 322 s
warmup + capture).

- **[reported]** **External corroboration (S-dgxspark-report):** a published DGX-Spark optimization
  writeup reports the same models in the same range — DeepSeek-V4-Flash-DSpark gen ~37 (c1) → 54 (c2)
  t/s (their "60–67" is the optimistic peak), MiMo-V2.5-NVFP4 gen ~34 (c1) → 63 (c2), Qwen3.5-122B-A10B-int4
  gen ~27/req @ c4/4k. Our numbers matching theirs is the evidence our fabric is *not* hit by the
  kernel-6.17 RoCE regression they flag (`[[wiki/multinode-tp-and-networking.md]]`). Their numbers are
  external, not independently reproduced here.

**4 sparkrun cross-node bugs fixed this run** (in `orchestration/infiniband.py` + `scripts/ray_head.sh`;
**revert on sparkrun upgrade**) — without them the TP=2 vLLM/Ray recipes hang at cross-node init:
1. vLLM ignores deprecated `HOST_IP` → set **`VLLM_HOST_IP=<fabric>`** (else `mq_connect_ip` = mgmt).
2. Forced global **`NCCL_IB_GID_INDEX=3`** → empty (per-NIC auto: idx3 NIC0 / idx5 NIC1).
3. **`NCCL_SOCKET_IFNAME`** listed mgmt NIC first → fabric NICs only.
4. **`ray_head.sh`** echoed the mgmt IP as `head_ip` (worker `ray start --address` → mgmt) → fabric.
See `[[wiki/multinode-tp-and-networking.md]]`.

## DiffusionGemma-26B-A4B — bf16 vs NVFP4 (2026-07-01, TP=2, `llama-benchy` pp512/tg128, 3 runs)

- **[proven]** First-party TP=2 GB10 measurements.

| Quant | prefill pp512 c1 | e2e TTFT | "decode" tg128 c1 | prefill c4 agg | "decode" c4 agg | Mem/node |
|---|---|---|---|---|---|---|
| NVFP4 (marlin MoE) | **287.5** t/s | ~1.8 s | 92.3 ± 29.8 | 282.5 | 262.3 | ~18 GB total |
| bf16 (triton MoE) | 159.2 t/s | ~3.4 s | 128.0 ± 0.0 | 251.1 | 223.0 | ~52 GB total |

> **[proven] Caveat — don't trust the decode column for block-diffusion.** DiffusionGemma denoises a
> whole 256-token canvas and emits in a burst, so `llama-benchy`'s per-token tg metric is an artifact
> (bf16's `128.0 ± 0.0` = it just returned the 128 requested tokens ~all at once; NVFP4's high
> variance is the same effect, noisier). The **trustworthy** comparisons are **prefill (NVFP4 ~1.8×
> faster)** and **memory (NVFP4 ~1/3)** — by those NVFP4 was the stronger serving option. We deploy
> **bf16** as the full-precision choice regardless (GB10 has no native FP4 → NVFP4 = weight-only
> marlin decompress). NVFP4 retired + weights deleted. See `[[wiki/models/diffusiongemma.md]]`.
> Bench needs `--skip-coherence` (short-probe empty-output quirk).

## Concurrency (where measured)

- **[proven]** **Holo NVFP4 text** (256 tok/req, util 0.60): 1→76, 8→295, 32→575, 64→758, **128→899
  tok/s (ceiling, 11.8×)**. Saturates ~96–128 (compute-bound Marlin FP4). Interactive op point 32–64.
- **[proven]** **Holo NVFP4 vision** (1280×800 screenshot/req, grounding): peaks ~1.69 steps/s @ 32;
  prefill-bound; interactive sweet spot **4–8 concurrent agents**.
- **[proven]** **Holo thinking ON vs OFF** (conc 8): 1.13 → 4.74 steps/s (**4.2×**), 190 → 14 output
  tok/step, no grounding loss. See `[[wiki/models/holo-3.1.md]]`.
- **[proven]** **MiniMax-M2.7-abliterated AWQ decode scaling** (tg128, EP=2): c1 **23.9** (per-req
  23.9), c2 35.6 (18.5/req), c4 53.6 (14.1/req), c8 **77.0** (10.2/req). With MTP/spec the peak is 25 /
  40 / 61 / 96. Knee ~c4. (NVFP4 c8 aggregate ~58, peak 77 — consistently below AWQ.)

> **[proven] Durable finding (S-minimax-sweeps):** for MiniMax-M2.7 on GB10, **AWQ-4bit decodes ~1.4×
> faster single-stream than NVFP4** (~24 vs ~16.5 tok/s) despite both being ~4-bit — the AWQ/Marlin
> path is more decode-efficient than NVFP4 FlashInfer-CUTLASS for this MoE here, while NVFP4 wins
> prefill. This is *why* the AWQ checkpoint is the durable default. Note this does **not** contradict
> "fewer bytes = faster" (`[[wiki/quantization-on-gb10.md]]`): AWQ and NVFP4 move ~equal weight bytes,
> so the delta is kernel efficiency, not bandwidth. Pick quant by *measuring decode on the actual
> model* — don't assume NVFP4 > AWQ. (Both EP=2 cross-node; ignore the bogus 303410 tok/s and 9.4 ms
> prefill rows = prefix-cache-inflated.)

## Method notes
Single-stream code/prose at temp 0 (first-party bench scripts; Atlas: llama-benchy `spark-arena-v2`,
prefill 2048 / depth 0 / concurrency 1). Vendor-claimed numbers tagged "(vendor)" — not independently
reproduced. `max_model_len` does **not** affect single-stream decode (only KV pool). First-shape
cudagraph capture inflates first-run TTFT — re-run warm.

## MiMo-V2.5-NVFP4 + DFlash spec-decode (2026-07-06, custom_class, TP=2, S-dflash-nvfp4)

- **[proven]** Decode tok/s single-stream, temp 0. Spec accept is **workload-dependent** (structured
  >> prose):

| workload | tok/s | accept tok/block | note |
|---|---|---|---|
| math | 38.5 | 3.29 | short ctx, num_spec=5 |
| json | 42.2 | 3.23 | short ctx, num_spec=5 |
| code | 29.9 | 1.87 | short ctx, num_spec=5 |
| prose | 23.7 | 1.31 | short ctx, num_spec=5 |
| prose @ 100k depth | 8.8 | 0.86 | llama-benchy pp256/tg128, War&Peace |
| prose @ 200k depth | 4.3 | 0.87 | target-attention-bound (O(ctx)) |

- **[proven]** **Tuning:** `num_speculative_tokens=5` beats 7 by +15-20% tok/s (drafter block_size=8;
  the accept profile decays so positions 6-7 add ~2% accept but cost 2 verify positions/step).
  Deep-context decode is target-bound — nvfp4 diffkv attention is ~2x slower than fp8 at 200k (cf
  HeNryous fp8 200k@10 vs our 4.3); that gap is KV-dtype/kernel, not the drafter.
- **[proven]** Deep context: coherent + correct needle recall verified at **18k/45k/89k** (real varied
  content). Full **1M** context fits at GMU 0.88 (2.1M-token pool); 1M *prompt* prefill is ~30 min
  (O(context)) — 256k+ is the practical range. Drafter is eager PyTorch (per-request loop) — tok/s has
  headroom vs a fused/cudagraph path.
- **[reported]** cf forum HeNryous fp8: 100K@12 tok/s (fp8 decodes a bit faster than nvfp4) — external
  forum number, origin HeNryous.
