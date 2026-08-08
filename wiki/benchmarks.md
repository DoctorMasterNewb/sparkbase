# Benchmarks (GB10 / DGX Spark)

> **area:** benchmarks
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-m3-20tps, S-nemotron-rpc, S-mimo-doc, S-minimax-sweeps, S-swapper-sweep, S-dgxspark-report, S-diffusiongemma, S-forum-dsv4-flash, S-forum-dsv4-dspark, S-forum-glm52-4x, S-forum-mimo-2x, S-forum-mimo-3x, S-forum-m3-llamacpp-2x, S-forum-m3-awq-4x, S-forum-mxfp4-patches, S-forum-qwen122, S-forum-mimo-dflash-22-67, S-forum-glm47-full-2x, S-forum-ds4f-4x-vllm, S-forum-nemotron-super-mtp, S-forum-nemotron-ultra-4x, S-forum-m25-sglang-4x, S-forum-glm47-rdma, S-forum-nemotron-2node, S-forum-dsv4-dspark-eugr, S-forum-4node-qrs812, S-forum-glm52-3x-aqlm, S-forum-comfyui-triplany, S-forum-dsv4-0731-bench, S-forum-dsv4-0731-dspark-loader, S-forum-macaron-v1-tall, S-forum-minimax-h3-comfyui, S-forum-dsv4-0731-ds4-cuda, S-forum-laguna-modelopt, S-forum-sparkring, S-forum-dsv4-llamacpp-fan, S-forum-kimi-k3-coder-reap
> **updated:** 2026-08-08

Single-stream decode unless noted. All on the 2× GB10 pair unless noted (single-node). Numbers
anchor the rules on
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
| Laguna-S-2.1-NVFP4 | NVFP4 (W4A4) + DFlash spec=7 | vLLM 0.25.1 + FlashInfer nightly | 1 | **22.6** (peak 32.7) | 256k | 69.3 GiB | 117.6B MoE, 8.5B active, SWA; decode flat across depths; DFlash low accept on prose | S-laguna-v251-bench |
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

## Forum-reported benchmarks (2026-07-08 ingest)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Tagged `[forum]` to distinguish from the proven rows above.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| DeepSeek-V4-Flash (official) | FP8 (E4M3 128×128 block) | vLLM (mp, no-ray) | 2 (TP=2) | ~44 | 200K | MTP nst=2, block-size 256, cudagraph FULL_AND_PIECEWISE | S-forum-dsv4-flash ||
|| DeepSeek-V4-Flash-DSpark | NVFP4 (`nvfp4_ds_mla` KV) | vLLM+DSpark self-spec | 2 (TP=2) | ~60-67 (code) / ~40 (mixed) | 1M | c=6 ~182 tok/s agg; c=16@200K ~315 agg | S-forum-dsv4-dspark ||
|| GLM-5.2 (744B/40B MoE) | AWQ-INT4 + 15% expert prune | vLLM | 4 (TP=4) | ~22 | 256K | MTP gave biggest uplift; cudagraph only ~3% | S-forum-glm52-4x ||
|| MiMo-V2.5-NVFP4 | NVFP4 + fp8 KV | vLLM (Ray) | 2 (TP=2) | 36-42 | 131K | Q&A 37, code 40, JSON 42, math 35 (run 2) | S-forum-mimo-2x ||
|| MiMo-V2.5-Omni | NVFP4 | vLLM (Ray) | 3 (TP=3) | 38.8 (eff 35.1) | 1M | Virtual-head padding; thinking-OFF > ON | S-forum-mimo-3x ||
|| MiniMax-M3 426B | UD-IQ4_XS GGUF | llama.cpp RPC | 2 (layer-split) | ~10.7 | 65K | ~590 prefill; tool-calling via hybrid template | S-forum-m3-llamacpp-2x ||
|| MiniMax-M3-AWQ | AWQ 4-bit + fp8 KV | vLLM | 4 (TP=4) | ~30 | 262K | Adaptive reasoning | S-forum-m3-awq-4x ||
|| Qwen3.5-35B-A3B | MXFP4 (patched) | vLLM 0.17.0 (patched) | 2 (TP=2) | 70.68 | — | +65% over vanilla 42.85 | S-forum-mxfp4-patches ||
|| gpt-oss-120b | MXFP4 (patched) | vLLM 0.17.0 (patched) | 2 (TP=2) | 80.88 | — | +56% over vanilla 51.82 | S-forum-mxfp4-patches ||
|| Qwen3.5-122B-A10B | int4 | vLLM | 1 | up to 51 | — | eugr patches + quick-start | S-forum-qwen122 ||
|| Qwen3.5-122B-A10B | int4 + DFlash n=12 | vLLM 0.23 (patched) | 1 | ~81 (agent) / 59 (e2e) | — | block-spec decode, accept len ~8.3 | S-forum-dflash-qwen122 ||
|| Qwen3-Next-80B | native (Atlas) | Atlas | 1 | 82 | — | 2.8× vLLM, no spec decode, Rust+CUDA | S-forum-atlas ||
|| DeepSeek-V4-Flash Q2 | Q2 GGUF | ds4 (DwarfStar 4) | 1 | ~28 | — | custom CUDA engine, 81 GiB | S-forum-ds4-cuda ||
|| Hy3 (295B/21B MoE) | NVFP4-W4A16 | vLLM 0.23.1 (Ray) | 2 (TP=2) | 21.8 / 59.7 agg@c6 | 128K | MTP nst=1 (pos-2 only 20%), enforce-eager wins | S-forum-hy3 ||
|| GLM-5.2 NVFP4 | NVFP4 | vLLM | 4 (TP=4) | 24 | 128K | MTP4 fix: config plumbing bug, accept ~0.84 pos-4 | S-forum-glm52-mtp-fix ||
|| GLM-5.2 AWQ-INT4 (pruned) | AWQ-INT4 + 15% prune | vLLM | 4 (TP=4) | 22 | 256K | MTP biggest uplift, cudagraph ~3% | S-forum-glm52-4x ||
|| GLM-5.2 1-bit | UD-IQ1_S GGUF | llama.cpp RPC | 2 (layer) | 8 | 256K | toy experiment, 1-bit quant | S-forum-glm52-1bit ||
|| MiniMax-M3-AWQ | AWQ-INT4 + fp8 KV | vLLM (Ray) | 4 (TP=4) | 33 | 262K | EAGLE3, 5 GB10 build fixes (CUDA 13.0 mismatch) | S-forum-m3-awq-tp4 ||
|| MiniMax-M3-AWQ | AWQ-INT4 + nvfp4 KV | vLLM | 4 (TP=4) | 25 | 1M | nvfp4 KV inline-dequant fused, 1M pool | S-forum-m3-awq-1m ||
|| MiniMax-M3-MXFP4 | MXFP4 + bf16 KV | vLLM nightly | 4 (TP=4) | 35 | 262K | EAGLE3 k=2, no fp8 KV (crashes), ~70 tok/s@c5 | S-forum-m3-mxfp4-4x ||
|| MiniMax-M3-W4A16-GPTQ | W4A16 + nvfp4 KV | vLLM b12x | 2 (TP=2) | 33 (+vision) | 113K | OCR-grade multimodal, vision reproduced | S-forum-m3-vision-b12x ||
| Qwen3.6-27B (dense) | FP8 + MTP nst=3 | vLLM (spark-vllm-docker) | 1 | 15.2 | 32K | 7.8 baseline → 1.94× with MTP; bandwidth-bound ~10 tok/s ceiling | S-forum-qwen36-27b-fp8 ||
| Gemma-4-31B-IT | NVFP4 + MTP=7 (assistant drafter) | vLLM (eugr fork) | 2 (TP=2) | 24.78 (peak) / 14.1 (tg1024) | auto | fp8 KV, MTP=7 optimal for 31B, MTP=4 for 26B-A4B | S-forum-gemma4-assistant ||
| GLM-4.7 (355B full) | NVFP4 | vLLM (spark-vllm-docker) | 2 (TP=2) | 17.5 | 64K | NCCL_NET_GDR_LEVEL=0 mandatory; --no-ray; 4 documented walls | S-forum-glm47-full-2x ||
| DeepSeek-V4-Flash | FP8 + MTP | vLLM (jasl fork) | 4 (TP=4) | 49.4–54.4 (single) / 180 (n8 agg) | 384K | NCCL 2.30.4 critical (2.28.9 wedges); sm12x_deep_gemm_fallbacks.py | S-forum-ds4f-4x-vllm ||
| Nemotron-3-Super-120B | NVFP4 + MTP | SGLang 0.5.13-dev | 4 (TP=4) | 1.70× single-stream (over no-spec) | 524K | accept_len ≈2.7; 3/4 depth beats NVIDIA cookbook 5/5; Mamba state pool limits single-node | S-forum-nemotron-super-mtp ||
| Nemotron-3-Ultra-550B | NVFP4 (modelopt_mixed) | SGLang 0.5.12 | 4 (TP=4, EP=4) | 42–43 (n8 peak) / 5.3 (per-req n8) | 512K | 83.7 GB/GPU weights; LatentMoE 512+1 experts, Mamba2+MoE+attn hybrid | S-forum-nemotron-ultra-4x ||
| MiniMax-M2.5 | NVFP4 | SGLang | 4 (TP=4, EP=4) | 25.5 (single) / 124 (n8 agg) | — | MAX_JOBS=1 fixes CUTLASS MoE compile OOM; RDMA enabled | S-forum-m25-sglang-4x ||
| GLM-4.7-FP8 | FP8 | SGLang | 4 (TP=4) | 25.1 (RDMA) / 8.2 (socket) | — | 2.5× speedup just from RDMA enable; SGLang container needs --device=/dev/infiniband | S-forum-glm47-rdma ||
| GLM-5.2 (744B) | IQ4_XS GGUF | llama.cpp RPC | 4 | 6.28 (C=1) / 9.07 (C=4 agg) | 1M | DSA active; ngram self-spec →24 tok/s structured; Q4_K_S has more headroom | S-forum-glm52-iq4xs-4x ||
| Qwen3.5-397B-A17B | NVFP4 + MTP | SGLang 0.5.10 | 4 (TP=4) | 40.0 (n1, MTP) / 110.9 (n8) | 524K | MTP +86% single-stream; cutlass MoE + triton attn + fi_cutlass FP4 + CG on wins | S-forum-roce-397b-mtp ||
| Gemma-4-31B (dense) | BF16 + MTP (assistant) | SGLang 0.5.11 | 4 (TP=4) | 26.68 (n1, MTP=6) / 153.24 (n8) | 262K | FROZEN_KV_MTP drafter; +154% @ n1, +80% @ n8 over baseline | S-forum-gemma4-mtp-4x ||
| Qwen3.5-122B-A10B | NVFP4 (RedHatAI) | vLLM | 1 | 16.15 | 262K | Quality close to FP16; moe-backend flashinfer_cutlass | S-forum-qwen122-nvfp4-redhat ||
| Qwen3.5-122B-A10B | NVFP4 (community) | vLLM | 1 | — | — | 234GB→75.6GB, fits 128GB; DeltaNet+vision, routers/lm_head kept BF16 | S-forum-qwen122-nvfp4-quant ||
| Nemotron-3-Super-120B | NVFP4 | vLLM TP=2 | 2 | 24 | — | ABI fix for cu130/cu132 mismatch in Dockerfile (cu132 wheel + cu130 PyTorch) | S-forum-nemotron-super-abi ||
| DeepSeek-V4-Flash | hybrid 2-bit (IQ2_XXS+Q2_K+FP8) | vLLM (custom) | 1 | — | — | antirez MLX recipe ported; ~85 GiB, coherent output on single Spark | S-forum-ds4f-hybrid-1x ||
|| Step-3.7-Flash | IQ4_XS GGUF | llama.cpp (stepfun fork) | 1 | 31 (short ctx) / 11 (max ctx) | 262K | Only path for Step-3.7 on Spark; prefill poor vs vLLM | S-forum-step37-llamacpp ||

## Forum-reported benchmarks (2026-07-10 ingest)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Measured with `llama-benchy` (context-depth sweep tool, see
`[[wiki/containers-and-tooling.md]]`).

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| MiniMax-M2.1-AWQ-4bit | AWQ 4-bit | vLLM | 2 | ~36 (tg32, c1) | 100K | pp2048 ~3544 t/s; degrades with depth (pp @ 8K ~2832); measured via llama-benchy | S-forum-llama-benchy ||
|| GLM-4.7-Flash-AWQ-4bit | AWQ 4-bit | vLLM | 1 | ~41.75 (tg32, c1) | 202K | pp2048 ~5326 t/s c1; c2 tg32 agg 73.74 (37.38/req); c10 tg32 agg 87.65 (15.33/req); KV cache 1.24M tokens, util 0.7 | S-forum-llama-benchy ||
||| FLUX.2-dev (image gen) | NVFP4 W4A4 (torchao) | torchao/diffusers | 1 | — | — | 28 steps @ 1024²: ~45s NVFP4 vs ~2.3 min BF16 (~3×); ~66 GB vs ~112 GB VRAM; edit ~1m51s vs ~4m20s (~2.3×) | S-forum-flux2-nvfp4-compute ||

## Forum-reported benchmarks (2026-07-10 ingest, Batch 5)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||| 
||| MiMo-V2.5 (native FP8) | FP8 (310B) | SGLang | 4 (TP=4) | 31.5 | 256K | EAGLE disabled (OOM); TTFT 0.46s; fp8 KV; tool eval 89/100; vision+audio+video | S-forum-mimo-sglang-4x ||
||| MiMo-V2.5-NVFP4 (renek recipe) | NVFP4 + fp8 KV | vLLM (Ray) | 2 (TP=2) | 30-33 (single) / 57-63 agg@c3 | 160K | enforce-eager, MTP=2 (86%/45% accept), util=0.89, triton_attn_diffkv; 38 tok/s claimed by tonyd615 (non-eager) | S-forum-mimo-2x-opt ||
||| MiniMax-M2.7-NVFP4 | NVFP4 (FlashInfer-CUTLASS) | vLLM | 2 (TP=2) | 24.12 (tg128) | 225K | FlashInfer-CUTLASS + throughput backend; no-Ray slightly better; CUTLASS baseline ~22 | S-forum-m27-recipe ||
||| MiniMax-M2.7-AWQ-4bit | AWQ 4-bit | vLLM | 2 (TP=2) | 39.4 (tg128) / 41.6 (tg32) | 196K | Clear decode winner — ~1.5× NVFP4; 3 independent reporters agree | S-forum-m27-recipe ||
||| MiniMax-M2.7 (Unsloth FP8) | FP8 | vLLM | 4 (TP=4) | 36–37 | — | No degradation vs NVFP4, slight increase; cache hit 53.6 @ 2 concurrent | S-forum-m27-recipe ||

## Forum-reported benchmarks (2026-07-13 ingest, Batch 11)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Measured on a 4-node cluster via CRS504 switch (100G). `llama-benchy` c=1, pp2048.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| DeepSeek-V4-Flash | FP8 KV, MTP k=2 | vLLM (aidendle B12X) | 4 (TP=4) | 52.0–53.6 (TG) | 512K | PP 2236–2452; no decode loss vs 200G; 100G link sufficient | S-forum-4node-crs504 ||
||| DeepSeek-V4-Flash | FP8 KV, MTP k=2 | vLLM (aidendle B12X) | 2 (TP=2) | 29.9–36.8 (TG) | 256K | PP 1612–2025; measured pre-4-node baseline | S-forum-4node-crs504 ||
||| MiniMax-M3-AWQ | AWQ-INT4 + bf16 KV + EAGLE | vLLM TP=4 | 4 (TP=4) | 27.7–35.4 (TG c=1) | 262K | PP 1684–2211; mns=4; 4-node CRS504 switch | S-forum-4node-crs504 ||

> **[conjecture]** 4-node CRS504 (100G switch) vs direct 200G: PP loss only 5–10%, decode
> unchanged. Measured traffic ~13 Gb/s — well below 100G. TP=4 DSV4-Flash decode 52–53.6 tok/s
> (vs TP=2's 29.9–36.8) shows near-linear scaling from 2→4 nodes. M3-AWQ+EAGLE on 4-node
> (28–35 tok/s) is consistent with existing [reported] M3-AWQ TP=4 benchmarks (S-forum-m3-awq-tp4
> 33 tok/s, S-forum-m3-awq-4x ~30 tok/s). (S-forum-4node-crs504)

## Forum-reported benchmarks (2026-07-14 ingest, Batch 12)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Measured with `llama-benchy` (MTP2 + fp8 KV + prefix cache, C=1 × 3 runs).

|||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
||||---|---|---|---|---|---|---|---|||
|||| DeepSeek-V4-Flash | MXFP4 MoE, fp8 KV, MTP2 | TokenSpeed `sm12x-stable` | 2 (TP=2) | 30.3–33.3 (tg128 peak) | 131K | PP 1979–2062 @ 8K-32K depth (+10–14% vs vLLM); KV 1.90M tokens (+25%); tool 45/45, GSM8K 0.96 | S-forum-tokenspeed |||
|||| DeepSeek-V4-Flash | MXFP4 MoE, fp8 KV, MTP2 | vLLM (jasl fork) | 2 (TP=2) | 41.3–45.3 (tg128 peak) | 131K | PP 1737–1866 @ 8K-32K depth; KV 1.52M tokens; decode ~30% faster than TokenSpeed | S-forum-tokenspeed |||

> **[conjecture]** TokenSpeed `sm12x-stable` vs jasl/vllm fork on the same 2× Spark pair: TokenSpeed
> leads cold-context prefill by ~10–14% but decode is behind ~70–74%. The CUTLASS MoE backend that
> wins prefill has a weaker small-M decode GEMM; a hybrid CUTLASS-prefill + Triton-decode path is in
> progress. TokenSpeed also fits +25% more KV cache (1.90M vs 1.52M tokens) and has zero tool-calling
> HTTP 500s. Both are single-source (jasl). (S-forum-tokenspeed)

## Forum-reported benchmarks (2026-07-15 ingest, Batch 14)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Single-node DGX Spark, vLLM 0.24, MTP enabled.

||||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
|||||---|---|---|---|---|---|---|---|||
||||| Qwen3.6-35B-A3B (nvidia) | NVFP4 + MTP=2, fp8 KV | vLLM 0.24 | 1 | ~90 (c=1) / ~420 (c=16 agg) | 262K | nvidia/Qwen3.6-35B-A3B-NVFP4; Marlin MoE; baseline for Unsloth comparison | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (Unsloth) | NVFP4 + MTP=2, fp8 KV | vLLM 0.24 | 1 | ~75 (c=1) / ~410 (c=16 agg) | 262K | unsloth/Qwen3.6-35B-A3B-NVFP4; ~15% slower than nvidia; gap narrows at high concurrency | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (nvidia) | NVFP4 + MTP, fp8 KV | vLLM 0.24 | 1 | 103.4 avg (4 workloads) | — | hedelyuk.alexandr benchmark: short_ok 107.2, coding 112.3, reasoning 93.2, agentic 100.8 | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (Unsloth) | NVFP4 + MTP, fp8 KV | vLLM 0.24 | 1 | 87.6 avg (4 workloads) | — | hedelyuk.alexandr: short_ok 92.9, coding 95.7, reasoning 77.3, agentic 84.6; −15.2% avg | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (Unsloth-Fast) | NVFP4 + MTP, fp8 KV | vLLM 0.24 | 1 | 97 (tuned) / 75 (initial) | 262K | unsloth/Qwen3.6-35B-A3B-NVFP4-Fast; TheAwakenOne tuned recipe; still behind nvidia | S-forum-unsloth-qwen36 |||

> **[reported]** Unsloth NVFP4 is consistently ~15% slower than nvidia NVFP4 on GB10 across 3
> independent benchmarks (hedelyuk.alexandr, J-R, TheAwakenOne). The "2.5× faster" claim is B200-only.
> At c=16 the gap narrows to ~2% (420 vs 410 tok/s aggregate). (S-forum-unsloth-qwen36)

## Forum-reported benchmarks (2026-07-15 ingest, Batch 15)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Qwen3.5-397B-A17B | FP8 | — | 8 (TP=8) | 31–35 | — | largest reported DGX Spark cluster; MoE gains flatten past TP=4; no config details provided | S-forum-qwen397-arch ||

> **[conjecture]** Single forum reference (raphael.amorim) citing the 8× GB10 cluster result.
> No engine, flags, or configuration details provided — treat as a data point only.

## Forum-reported benchmarks (2026-07-16 ingest, Batch 16)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| GLM-5.2 (744B/40B MoE) | int4 MoE + int8 MTP | Colibri (expert streaming) | 1 | 2.39 (full top-8) / 3.33 (CACHE_ROUTE) | short | Experts streamed from disk; O_DIRECT 9.69 GB/s; 82-97% expert cache hit; RSS 76-78.5 GB; `COLI_CUDA_UNIFIED=1`; single-Spark 744B | S-forum-colibri-glm52 ||

> **[conjecture]** Colibri is the first reported engine to run a 744B MoE on a single 121 GB Spark,
> streaming experts from disk (only hot experts cached via LRU/pin). 2.4-3.3 tok/s is very slow but
> coherent — the streaming-from-disk approach is fundamentally different from multi-node TP. Attention
> dominates the profile (6.16s of 18s), not disk I/O. Experimental CACHE_ROUTE (cache-aware routing,
> ~14% expert substitution) raises hit 82→97% and tok/s 2.4→3.3; not upstream default. (S-forum-colibri-glm52)

## Forum-reported benchmarks (2026-07-17 ingest, Batch 18)

All rows below are **[conjecture]** — single-source community-reported numbers (two forum users in one
thread, not independent threads), not first-party.

||||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
|||||---|---|---|---|---|---|---|---|||
||||| GLM-5.2 (744B/40B MoE) | Int4-Int8 mix (QuantTrio GLM-5.2-Int4-Int8Mix) | vLLM v16-unified + b12x W4A8 | 8 (TP=8, DCP=1) | 33–54 avg (33–39 prose, 40–55 code, peak 54.5–58) | 200K | ~1,200 t/s prefill (1,000→1,200 vs older branch); TTFR 1.7s→198s @ 0–200K depth; 2 conc → 50 (prose) / 60–70 (code); Snake game 54.16 tok/s; tool-eval 91/100 | S-forum-glm52-8x |||
||||| GLM-5.2 (744B/40B MoE) | Int4-Int8 mix | vLLM v16-unified | 8 (TP=4+PP2, experimental) | ~12 | — | ~1,800 t/s prefill but MTP acceptance collapses to ~8% → decode drops to ~12 t/s; NOT viable | S-forum-glm52-8x |||
||||| GLM-5.2 (744B/40B MoE) | Int4-Int8 mix | vLLM v16-unified (decode-aware scheduler) | 8 (TP=8, DCP=4, MTP3) | ~2.74 (under prefill pressure) / 26 (post-prefill) | 320K (×10 seq) | DCP4 gives 3.2M KV tokens; decode starves to ~0.0–0.2 tok/s during prefill; scheduler patch ENABLE_DECODE_AWARE_PREFILL=1 limits stalls to ~1.6s | S-forum-glm52-8x |||

> **[conjecture]** **GLM-5.2-Int4-Int8Mix on 8× GB10 is the largest reported DGX Spark cluster run**
> (S-forum-glm52-8x, ciprianveg + penguinchang, single thread). Key numbers from the OP benchmark
> (llama-benchy, tg=1500, single stream):
> - **Prefill stays >1,000 t/s all the way to 200K context** (1,211 at depth 0, 1,019 at 200K).
>   The v16-unified branch (local-inference-lab/vllm @ 5dffea8, branch
>   `codex/fathomless-firmament-v16-unified-20260712`) is the single biggest prefill lever — older
>   branch capped ~1,000, v16 climbs to ~1,200.
> - **Single-stream avg decode: 33–39 t/s on coherent prose (stable across 0–200K context),
>   40–55 t/s on coding/structured** (peak 53.5–58 t/s). Two concurrent: ~50 (prose) / 60–70 (code).
>   Snake game generation (temp=0, thinking=off): 54.16 tok/s.
> - **Stack:** vLLM v16-unified fork + b12x W4A8 MoE (lukealonso/b12x @ 97b3d64, unified SM120
>   sparse MLA + PCIe DCP collectives) + DCP1 patches from CosmicRaisins/glm-5.2-gb10. CUDA 13.2.0,
>   PyTorch 2.11.0, NCCL 2.30.4 (custom aarch64), transformers ≥5.0 (--tf5 build flag), prebuilt
>   sm_121 FlashInfer wheels.
> - **DCP1 knobs (3 vs CosmicRaisins base):** `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` (keeps B12X
>   indexer path consistent), `draft_tensor_parallel_size=1` (unsharded drafter avoids TP8
>   collectives on every draft step), `NCCL_BUFFSIZE=16777216` (16 MB, up from 8 MB — bigger
>   NCCL buffer for gen speed at high context; 8 MB starts bottlenecking allreduce on long-context
>   decode).
> - **TP4+PP2 is not viable for MTP:** prefill climbs to ~1,800 t/s but MTP acceptance collapses to
>   ~8% on the pipeline split → decode drops to ~12 t/s. Staying on TP8+PP1 for production. (OP)
> - **DCP4 decode starvation (penguinchang):** on TP8 DCP4 with MTP3, concurrent prefill requests
>   starve decode to ~0.0–0.2 tok/s until prefill completes. A custom "decode-aware prefill"
>   scheduler (ENABLE_DECODE_AWARE_PREFILL=1, DECODE_PREFILL_TOKEN_BUDGET=1024,
>   IDLE_PREFILL_TOKEN_BUDGET=16384, MAX_LONG_PREFILLS_PER_STEP=1) limits stalls to ~1.6s max
>   but decode still falls to ~2.74 tok/s under pressure. DCP1 would get ~30% faster prefill and
>   ~60% faster gen per the OP, but DCP4 enables 320K×10 context (3.2M KV tokens).
> - **Four production patches:** 01 (DCP config → draft model, prevents MTP collapse under DCP>1),
>   03 (quantized NextN draft token mapping — without it quantized drafts silently build
>   unquantized and MTP acceptance collapses), 04 (DeepSeekMTP SupportsPP + stale topk_indices_buffer
>   in flashinfer SM120 sparse MLA PR#46994 + MTP embed_tokens loading under PP), 06 (b12x stale
>   topk buffer PR#46994 Fix #4 — without it _maybe_share_lm_head swaps the indexer's buffer but the
>   backend keeps a stale ref → garbage DSA attention and ~30% acceptance instead of ~85%).
>
> This corroborates: (1) NCCL 2.30.4 mandatory on multi-node (S-forum-ds4f-4x-vllm, S-forum-tokenspeed),
> (2) MoE gains flatten / interconnect-bound at scale (S-forum-qwen397-arch), (3) pipeline parallelism
> is latency-sensitive and can wreck MTP on Spark (S-forum-2d-parallel). All numbers are from one
> thread with two active users — `[conjecture]`, not `[reported]` (the two posters are in the same
> thread and not independent).

## Forum-reported benchmarks (2026-07-16 ingest, Batch 17)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Numbers from a meta-analysis thread (S-forum-nvfp4-broken) citing multiple independent
forum sources, plus an abliterated model variant (S-forum-dsv4-abliterated).

||||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
|||||---|---|---|---|---|---|---|---|||
||||| Llama-3.3-70B-Instruct-NVFP4 | NVFP4 | TensorRT-LLM | 1 | 5 (reporter 1) / 2.5 (reporter 2) | — | NVIDIA's own NVFP4 model on NVIDIA's TRT-LLM; slower than GGUF Q4_K_M via LM Studio (4.6-4.9 tok/s) on same Spark | S-forum-nvfp4-broken |||
||||| Llama-3.3-70B-Instruct | GGUF Q4_K_M | LM Studio (llama.cpp) | 1 | 4.6–4.9 | — | Non-NVIDIA quant on non-NVIDIA tooling beats NVIDIA NVFP4 on TRT-LLM | S-forum-nvfp4-broken |||
||||| Nemotron-3-Super-120B-A12B | NVFP4 | vLLM | 1 | 19–22 | — | 42-48% of theoretical bandwidth ceiling (~45 tok/s); 12B active @ 0.5 bytes = ~6 GB/token vs 273 GB/s | S-forum-nvfp4-broken |||
||||| Nemotron-3-Super-120B-A12B | NVFP4 | vLLM | 2 (TP=2) | 24 | — | ~200 GB/s cluster bandwidth; 71% of ~34 tok/s theoretical at that BW | S-forum-nvfp4-broken |||
||||| DeepSeek-V4-Flash-DSpark-Abliterated | NVFP4 (DSpark recipe) | vLLM (DSpark) | 2 (TP=2) | 50–60 | — | Abliterated (uncensored) variant; fork of DS4 DSpark recipe with model swapped in | S-forum-dsv4-abliterated |||

> **[reported]** The NVFP4 meta-analysis (S-forum-nvfp4-broken) aggregates multiple independent forum
> measurements confirming that NVFP4 on GB10 achieves only 42–48% of the bandwidth-limited theoretical
> ceiling — a software/kernel gap, not a hardware limitation. TRT-LLM NVFP4 (NVIDIA's own stack) is
> slower than GGUF Q4_K_M (community tooling) for the same 70B model. Multiple sources agree.

## Forum-reported benchmarks (2026-07-20 ingest, Batch 24)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| GLM-5.2 (744B/40B MoE) | (unspecified, likely Int4-family) | vLLM + b12x | 6 (TP=6) | ~30 (single-stream) | — | 6× GB10 via MikroTik CRS812 (768 GB); b12x enables non-power-of-2 TP; cluster 800-1180 W peak; no config/YAML shared | S-forum-6x-cluster ||

> **[conjecture]** GLM-5.2 at TP=6 (~30 tok/s) sits between the TP=4 (~22-24 tok/s,
> S-forum-glm52-4x / S-forum-glm52-mtp-fix) and TP=8 (33-54 tok/s, S-forum-glm52-8x) results —
> consistent with sublinear scaling as interconnect overhead grows with node count. Quant format
> unspecified by the poster; "b12x" usage and ~30 tok/s range are consistent with the Int4-Int8 mix
> used in the 8× run. Single source (mclenithan), no benchmarking methodology described.

## Forum-reported benchmarks (2026-07-20 ingest, Batch 25)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Inkling (Thinking Machines, 975B/41B-active MoE) | NVFP4 | vLLM (forked, 12 patches) | 8 (TP=8) | 25 (c1, short ~100 tok) | short | cudagraphs ON; MTP k=1 stuck (60% draft accept); LAMPORT_RS_SCONV=0 for RoCE; recipe at blockmos/inkling-sparks-gb10 | S-forum-inkling-nvfp4 ||
|| Inkling | NVFP4 | vLLM (forked) | 8 (TP=8) | 27 (c1, short, MTP k=2) | short | MTP k=2 adds ~2 tok/s on short context | S-forum-inkling-nvfp4 ||
|| Inkling | NVFP4 | vLLM (forked) | 8 (TP=8) | 13.5 (c1, 2048 ctx) | 2048 | long-context cliff: paged-KV absent in tml_fa4 Sm120 → O(ctx) KV regather per token | S-forum-inkling-nvfp4 ||
|| Inkling | NVFP4 | vLLM (forked) | 8 (TP=8) | 80 (c8 total, short) / 193 (c32 total, short) | short | aggregate scales on short context; 24 tok/s aggregate ceiling at real context | S-forum-inkling-nvfp4 ||
|| Inkling (prefill) | NVFP4 | vLLM (forked) | 8 (TP=8) | ~1,400 (pp2048) / up to 2,711 (throughput cfg) | 2048 | prefill "higher than we ever got M3" | S-forum-inkling-nvfp4 ||

> **[conjecture]** Inkling NVFP4 on 8× Spark shows a steep long-context decode cliff: 25 tok/s
> (c1, ~100 tok) → 13.5 tok/s (c1, 2048 tok). The cliff is because the `tml_fa4` Sm120/Sm121 cute
> attention path has no paged-KV — the workaround re-gathers the whole KV history every decode step
> (O(ctx)/token), capping aggregate at ~24 tok/s at real context regardless of concurrency. NVFP4
> itself is clean (no dtype fallbacks). MTP stuck at k=1 (60% draft acceptance). Prefill is strong
> (1,400–2,711 tok/s). Single source (greg190), public repo + 12 patches + filed vllm#49049. The OP
> parked Inkling in favor of M3 (42 tok/s single-user, scales with concurrency). See
> `[[wiki/models/inkling.md]]`.

## Forum-reported benchmarks (2026-07-21 ingest, Batch 26)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Qwen3.5-397B-A17B | int4-AutoRound | vLLM (spark-vllm-docker) | 3 (PP) | 12–14.4 (tg32) | 32K+ | 3-node full mesh PP; decode ~single-node speed per eugr; PP+MTP not supported | S-forum-3node-mesh ||
|| Qwen3.5-397B-A17B (prefill) | int4-AutoRound | vLLM (spark-vllm-docker) | 3 (PP) | 912–1242 (pp2048) | 32K+ | prefill scales with depth: 912 @d0, 1242 @d8192, 1070 @d32768 | S-forum-3node-mesh ||

> **[conjecture]** Qwen3.5-397B-A17B-int4-AutoRound on 3-node pipeline-parallel (chunkai721 via
> llama-benchy v0.3.5). Decode ~12–14.4 tok/s across context depths 0–32768, confirming 3-node PP
> is ~single-node speed. Prefill 912–1242 tok/s. 3-node full-mesh CX-7 topology (no switch).
> `gpu_memory_utilization: 0.8` (0.85 causes silent worker death). Single source → [conjecture].

## Forum-reported benchmarks (2026-07-22 ingest, Batch 29)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Qwen3.6-35B-A3B (MoE) | NVFP4 | vLLM (PP=6, TCP fallback) | 6 (PP) | ~21 (per-req, 20 concurrent) | — | 326 tok/s aggregate; NCCL_IB_DISABLE=1 TCP transport; dummy0 identity addresses | S-forum-6x-ring-rdma ||
|| Qwen3.6-35B-A3B (MoE) | NVFP4 | vLLM (PP=6, RDMA) | 6 (PP) | ~21 (per-req, 20 concurrent) | — | 349 tok/s aggregate; NCCL_IB_MERGE_NICS=0 + NCCL_IB_SUBNET_AWARE_ROUTING=1; ~7% faster than TCP | S-forum-6x-ring-rdma ||

> **[conjecture]** Qwen3.6-35B-A3B-NVFP4 on 6-node pipeline-parallel (alpaslan.erdag). 20
> concurrent requests, ~21 tok/s per request, 326 tok/s aggregate (TCP) / 349 tok/s (RDMA).
> The ~7% RDMA-vs-TCP gain is attributed to GPUDirect RDMA being unavailable on GB10 —
> both transports are host-staged, so RDMA saves only TCP protocol overhead. See
> `[[wiki/multinode-tp-and-networking.md]]` → Batch 29 for the full topology findings.

## Forum-reported benchmarks (2026-07-23 ingest, Batch 30)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|---|---|---|---|---|---|---|---|---|
|| Mistral-Small-4-119B | NVFP4 + fp8 KV | vLLM 0.17.2rc1 (spark-vllm-docker) | 1 | 33.2 (2K ctx) / 31.7 (32K) / 17.7 (60K) | 65536 | TRITON_MLA, FLASHINFER_CUTLASS MoE; 10 concurrent → 100.3 tok/s agg | S-forum-mistral-s4-119b ||
|| Mistral-Small-4-119B | NVFP4 + fp8 KV | vLLM 0.21.0 (native arm64) | 1 | 28.76 (peak 30.0) | 256K | --shm-size 4g (16g crashes); max-num-seqs 4; bench serve c=1 | S-forum-mistral-s4-119b ||
|| Mistral-Small-4-119B | NVFP4 + fp8 KV | vLLM (patched) | 1 | 28.0 sustained | — | MLA enabled, no VLLM_MLA_DISABLE; reasoning_effort patched | S-forum-mistral-s4-119b ||
|| Mistral-Small-4-119B | NVFP4 + fp8 KV | vLLM (llama-benchy tg32) | 1 | 30.18 (d0) / 24.12 (d16384) | — | cosinus; prefill 3188→2560 tok/s across depths | S-forum-mistral-s4-119b ||
|| Mistral-Small-4-119B | NVFP4 + fp8 KV | vLLM (llama-benchy tg32) | 1 | 28.84 (d0) / 16.65 (d32768) | — | tenari; ctx_pp 3922-6195 tok/s; prefill degrades with depth | S-forum-mistral-s4-119b ||
|| Qwen3.6-35B-A3B | FP8 + fp8 KV | vLLM (spark-vllm-docker, no-ray) | 2 (TP=2) | 75-80 | 262K | Cold TTFT 0.68s (5K) / 8.49s (81K); prefix cache 2nd-run TTFT 0.47s/0.99s; 200GbE RoCE MTU 9000 | S-forum-qwen36-fp8-2x ||

> **[reported]** Mistral Small 4 119B NVFP4 decode at ~28-33 tok/s is corroborated by 5 independent
> forum users (mrDragonFox, cosinus, tenari, 0rand, chuckchambersdev) across different vLLM versions
> (0.17.2rc1, 0.21.0) and benchmark tools (vLLM bench serve, llama-benchy). The model fits on a
> single Spark (~60 GB NVFP4). Eagle/MTP does not work. See `[[wiki/models/mistral-small-4.md]]`.
>
> **[conjecture]** Qwen3.6-35B-A3B-FP8 on 2× Spark (gary100) — 75-80 tok/s output, native FP8 (not
> NVFP4). Lower than the 142 tok/s proven on Atlas with NVFP4+MTP (different engine/quant/stack).
> Single source → [conjecture].

## Forum-reported benchmarks (2026-07-24 ingest, Batch 32)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Solar-Open2-250B (250B-A15B MoE) | INT4 | vLLM | 2 (TP=2) | ~15 (tg32) | 32K+ | pp2048 ~2227 tok/s; flat decode across depths (14.3-14.9 @ d0-32768); no MTP tested; prefill degrades with depth (2227→1876 @ d32768) | S-forum-solar-open2 ||

> **[conjecture]** Solar-Open2-250B (South Korean government-backed, 250B-A15B MoE) INT4 on 2× Spark
> (FoRWiS). Decode ~15 tok/s (tg32), flat across context depths to 32K — consistent with a
> bandwidth-bound 250B MoE at 4-bit (~125 GB weights across 2 nodes). Prefill ~2227 tok/s at d0,
> degrading to ~1876 at d32768. No MTP tested. NVFP4 quant also available but not tested on Spark.
> Single source → [conjecture].

## Forum-reported benchmarks (2026-07-25 ingest, Batch 35)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---|||
|| DeepSeek-V4-Flash | unquantized (bf16) | WoolyAI Multi-agent Stack | 2 (TP=2) | 21.15 (C1) / 55.99 aggregate (C4, 14/req) | — | no spec decode; prefill 1518-1533 tok/s; model-activation-wait 16s; PDF report attached, no launch command shared | S-forum-woolyai ||
|| Gemma-4-26B-A4B | unquantized (bf16) | WoolyAI Multi-agent Stack | 2 (TP=2) | 30.22 (C1) / 63.75 agg (C4, 15.94/req) | — | no spec decode; prefill 4579-4702 tok/s; model-activation-wait 6s | S-forum-woolyai ||
|| Nemotron-3-Nano-Omni-30B | NVFP4 | WoolyAI Multi-agent Stack | 2 (TP=2) | 39.42 (C1) / 90.83 agg (C4, 22.71/req) | — | no spec decode; prefill 2588-2607 tok/s; model-activation-wait 2s | S-forum-woolyai ||

> **[conjecture]** WoolyAI Private Multi-agent Inference Stack (manisha5) — a closed-source
> multi-model agentic workflow server for 2× DGX Spark with a scheduler that swaps resident models
> at safe boundaries (model-activation-wait 2-16s). The C4 (4-concurrent) aggregate decode numbers
> (56-91 tok/s) are the only headline figures; per-request decode is 14-23 tok/s. **No launch
> command, no repro recipe, no source code shared** — only a PDF report and a product URL. Community
> skepticism (mrDragonFox): at C1 the numbers are "slower than llama.cpp" and "1/4 of the speed on
> ds4"; entrpi notes the community runs the spark-optimized DSpark fork which is "far more
> performant" (35 tok/s single-Spark DSV4-Flash). The C4 aggregate scaling (2.6-2.9× over C1) is
> consistent with batch-amortized bandwidth-bound decode but doesn't exceed what vLLM/SGLang achieve
> at equivalent concurrency. Treat as vendor-reported, unverified. Single source → [conjecture].

## Announced / upcoming models (2026-07-25)

> **[conjecture]** **Ant Ling-3.0-Flash 124B-A5B** (S-forum-ling3-flash, entrpi): announced by Ant
> Group (Alibaba); a 124B-total / 5B-active MoE with **hybrid-linear attention** (KDA:MLA layers
> stacked 5:1 — KDA for fine-grained long-range memory, MLA for efficiency) and **1/64 expert
> activation**. 256K context native, scales to 1M. Benchmarks reportedly beat their prior 1T model
> across nearly all benchmarks. Weights not yet released — expected "soon, probably after Aug 3."
> **GB10 relevance:** A5B active at 124B total is an unusually low active ratio (~4%) — if NVFP4
> (4.5 bit ≈ 70 GB) or AutoRound INT4 fits on a single 121 GB Spark, this could be a strong
> single-Spark contender vs. Qwen3.5-122B-A10B (current single-Spark GOAT) and DeepSeek-V4-Flash.
> Community expectation (xkm121): "INT4 auto-round or NVFP4 of this model will make this a good new
> contender." No weights, no tok/s yet → [conjecture]. Queued for re-ingest when weights drop.

## Forum-reported benchmarks (2026-07-26 ingest, Batch 36)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| Solar-Open2-250B (250B-A15B MoE) | NVFP4 W4A4 (nota-ai) | vLLM v0.25.1 (UpstageAI fork) | 2 (TP=2, no-Ray) | 15.8 (c1, d0) / 15.4 (c1, d32k) | 262K | FP8 KV: 2,665,802 tok pool (10.17× concurrency); bf16 KV: ~1.33M tok; prefill 924-983 (d0) → 1711 sustained @ 33k; TTFT 523ms; util 0.90, max-num-seqs 4; VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass | S-forum-solar-open2-nvfp4 ||

> **[conjecture]** **Solar Open2 250B NVFP4 on 2× Spark — linear attention makes decode ~free with
> depth; FP8 KV is a capacity lever, not a speed lever** (S-forum-solar-open2-nvfp4, danielgbates).
> Two durable GB10-relevant findings from a single well-documented source:
>
> 1. **Hybrid linear attention dodges the KV-bandwidth wall.** Solar Open2 has 36 of 48 layers as KDA
>    linear attention (12 GQA / 36 KDA mix). Decode at 32k context depth is within ~2.5% of
>    empty-context speed (15.4 vs 15.8 tok/s). Every full-attention model on the same pair decays
>    hard with context (a 310B MoE with NVFP4 KV drops to ~9 tok/s by 100k). **Why it bites on Spark:**
>    the proven bandwidth-bound decode ceiling (~270 GB/s) means full-attention KV grows with context
>    → decode slows. Hybrid-linear architectures that don't materialize per-token KV sidestep this.
>    This is the same finding class as Nemotron-3 (Mamba-2 hybrid, S-sess-jun5) and Holo-3.1 (hybrid
>    linear+full attention) — now generalized to a 4th architecture. See
>    `[[wiki/attention-and-kv-cache.md]]`.
> 2. **FP8 KV on hybrid-linear models is a capacity lever, not a speed lever.** On full-attention
>    models, fp8 KV is a decode-speed lever at depth (~2× vs 4-bit KV elsewhere). On Solar it's
>    speed-neutral — only 12/48 layers touch KV, so attention is a thin slice of decode time. What
>    you get instead is 2× pool: 10.17× concurrency at 262k, or headroom to push max-model-len toward
>    the model's native 1M. vLLM handles the mixed page layout fine (pads mamba page size 0.38% to
>    keep mamba and attention pages equal). No quality regression observed.
>
> **Stack details:** vLLM v0.25.1 built from source for aarch64/sm121 (CUDA 13.2, torch 2.11).
> SolarOpen2 isn't upstream — lives in UpstageAI's vLLM fork (`v0.22.0-solar-open2`), x86-only
> wheels/docker. The fork drops onto v0.25.1 almost unchanged (3-line adaptation: v0.25 fuses the
> KDA decay gate into the kernel, so the model's forward passes raw `g1` instead of calling
> `fused_kda_gate` first). Model: `nota-ai/Solar-Open2-250B-Nota-NVFP4` — 250B MoE (320 experts, 8
> active, ~15B active/token), NVFP4 W4A4 (llm-compressor, group 16), 153 GB / 29 shards, ~77 GiB
> weights/node. **Gotcha:** `--logits-processors` wants `module.path:ClassName` (colon) — the dotted
> form dies at worker init with `ValueError: not enough values to unpack (expected 2, got 1)`.
> Coherence probes pass in both KV configs (temp-0 arithmetic + reasoning-split + tool calls).
>
> This is a second-source corroboration of the existing INT4 Solar-Open2 row (S-forum-solar-open2,
> Batch 32, ~15 tok/s INT4 on 2× Spark) — same decode rate, now characterized with the NVFP4 W4A4
> both are single-source → [conjecture]. The flat-with-depth
> finding is the load-bearing result for the KB.

## Forum-reported benchmarks (2026-07-27 ingest, Batch 37)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| Qwen3.5-122B-A10B | int4 (Intel AutoRound) | vLLM | 2 | ~65 (holds linearly over 100k) | 100K+ | vision tower needed; best balanced model on 2-Spark cluster | S-forum-qwen122-king ||
||| Qwen3.5-122B-A10B | fp8 | vLLM | 1 | ~35 (at best) | — | marlin/deepgemm, no flashinfer; no tool-call quality improvement over int4 | S-forum-qwen122-king ||
||| Qwen3.5-122B-A10B (hybrid int4-fp8) | int4-fp8 hybrid (blesyg) | vLLM v26 (patched) | 1 | 40+ (5 concurrent lanes) | 256K | sparkrun-recipes repo; KV cache + overhead optimization; unpublished vLLM v26 patches | S-forum-qwen122-king ||
||| DeepSeek-V4-Flash | DSpark (NVFP4) | vLLM+DSpark | 1 | 45-50 (c1) / 240 (c16 agg) | 1M | coherent to 1M tokens; strongest 2-Spark alternative to 122B | S-forum-qwen122-king ||

> **[reported]** **Qwen3.5-122B-A10B-int4 is the community consensus single-Spark daily driver** —
> 4 independent forum users (Styles01, Josephbreda, 0rand, Rerollingingenshitimpactsucks)
> confirm it as the "king model" for single-Spark use: largest capable model that fits in 121 GB
> at high context, with a usable vision tower and good tool-calling. This corroborates the
> existing S-forum-qwen122 finding (up to 51 tok/s on 1× Spark). The new numbers: AutoRound int4
> on 2× Spark ~65 tok/s holding linearly past 100K context (Josephbreda); FP8 on 1× ~35 tok/s
> (0rand); sparkrun-recipes patched vLLM v26 build achieves 5 concurrent lanes at 256K context
> with 40+ tok/s decode (Styles01). The AutoRound int4 loop tendency is flagged by
> Rerollingingenshitimpactsucks — NVFP4 variants may offer better fidelity. See
> `[[wiki/models/qwen.md]]` → "king model" section for the full findings.
>
> **[conjecture]** DSV4-Flash on single Spark: 45-50 tok/s, 240 tok/s at 16 concurrent, coherent
> to 1M tokens (0rand). Consistent with existing S-forum-dsv4-dspark numbers (~60-67 tok/s on
> 2× Spark). Single source → [conjecture].

## Forum-reported benchmarks (2026-07-27 ingest, Batch 38)

All rows below are **[conjecture]** — single-source community-reported numbers (multiple users in one
thread, not independent threads), not first-party.

|||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| GLM-5.2 (744B/40B MoE) | Hybrid FP8+NVFP4+MXFP4 | vLLM eldritch + b12x (production-hybrid-1.2) | 4 (TP=4+DCP4) | 20-25 (c1 prose) | 800K | ~800 pp tok/s @100k depth; custom NVFP4 KV cache; adaptive spec depth; 2 reporters in same thread | S-forum-glm52-hybrid ||
|| GLM-5.2 (744B/40B MoE) | Hybrid FP8+NVFP4+MXFP4 | vLLM eldritch + b12x (production-hybrid-1.2) | 4 (TP=4+DCP4) | 20.1 (c1, d0) / 19.8 (c1, d4096) / 18.6 (c1, d8192) | 262K | llama-benchy v0.4.0 pp2048 tg128; pp 1605 (c1 d0) / 887 (c1 d4096) / 933 (c1 d8192); c4 decode drops to 9.9-17.8 | S-forum-glm52-hybrid ||
|| GLM-5.2 (744B/40B MoE) | Hybrid+GPTQ (v3: MXFP4-Experts-GPTQ) | vLLM eldritch + b12x (production-hybrid-1.3) | 4 (TP=4+DCP4) | 20.6 (c1) | 262K | tool-eval 85/100; pp 2,814 tok/s; TTFT 10,828ms; GPTQ applied on top of MXFP4 experts | S-forum-glm52-hybrid ||
|| GLM-5.2 (744B/40B MoE) | NVFP4 (official NVIDIA) | vLLM | 4 (TP=4) | (not benchmarked) | — | ~115 GB/node weights, ~460 GB total (excl. 20 GB MTP off by default); util ~98%; "pushes right up against limits" | S-forum-glm52-hybrid ||

> **[conjecture]** **GLM-5.2 Hybrid FP8+NVFP4+MXFP4 on 4× Spark** (S-forum-glm52-hybrid,
> aidendle94 + CosmicRaisins + alexander.korolev.germany): a community hybrid-quant checkpoint
> mixing FP8 (attention/some layers), NVFP4, and MXFP4 (experts that would be FP3) — the first
> reported 3-way mixed-precision GLM-5.2 checkpoint on GB10. Decode ~20-25 tok/s (c1 prose),
> consistent with the existing TP=4 range (22-24 tok/s for AWQ-INT4 / NVFP4 in
> S-forum-glm52-4x / S-forum-glm52-mtp-fix). Prefill ~800 tok/s at 100k depth. The llama-benchy
> table shows decode degrading from 20.1 (c1, d0) to 18.6 (c1, d8192) and to 9.9 at c4/d8192 —
> the depth+concurrency interaction is sharp. Custom NVFP4 KV cache with scaling/calibration;
> adaptive speculative depth; Docker image `sparkrun-vllm-ds4-gb10:production-hybrid-1.2/1.3`.
> Two reporters agree on ~800 pp / ~20-25 tg but are in the same thread using the same image →
> stays [conjecture]. See `[[wiki/models/glm-5.2.md]]` for the full recipe, tool-eval-bench
> quality results, and the reasoning-parser / repetition-penalty root causes.

## Forum-reported benchmarks (2026-07-28 ingest, Batch 39)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-122B-A10B | int4-fp8 hybrid + fp8 KV + DFlash + int8 lm-head | vLLM v26 (patched, 3 patches) | 1 | 45.98 (tg128) | 256K | KV 1,372,342 tokens (2.6× bf16); concurrency 5.24× @ 256K; prefill 957 tok/s (+32%); first fp8 KV + DFlash on GB10 for hybrid quant | S-forum-qwen122-v26-dflash |
| Qwen3.5-122B-A10B | int4-fp8 hybrid + bf16 KV | vLLM 0.23 (aeon) | 1 | 50.2 (tg128) | 256K | KV 549K tokens; concurrency 2.09× @ 256K; prefill 726 tok/s; baseline for fp8 KV comparison | S-forum-qwen122-v26-dflash |
| Qwen3.5-35B-A3B (MoE) | NVFP4 (GGUF) | llama.cpp (official full-cuda13) | 1 | 72.28 (tg128) | — | s-batman/Agents-A1-NVFP4-MTP-GGUF; 19.84 GiB; pp512 2636.97 tok/s; --mmap 0 -fa 1; matches custom build | S-forum-llamacpp-fastest |
| GLM-5.2 (744B/40B MoE) | int4 MoE + fp8 (expert streaming) | SpeedyColibri (Rust) | 1 | ~4 (with fp8) / ~1 (initial) | short | Rust port of Colibri; proof-of-concept; target 8 tok/s on 2× Spark | S-forum-speedycolibri |
| gemma-4-26B-A4B (unsloth NVFP4) | NVFP4 + fp8 KV | vLLM | 1 | ~17 (per-stream, n=100 conc) | 65K | aggregate 159.77 tok/s output @100 reqs; TPOT 47.14 ms; Unsloth ~17% faster than nvidia | S-forum-gemma4-26b-bench |
| gemma-4-26B-A4B (nvidia NVFP4) | NVFP4 + fp8 KV | vLLM | 1 | ~17 (per-stream, n=100 conc) | 65K | aggregate 128.21 tok/s output @100 reqs; TPOT 58.67 ms; ~6-7× slower than RTX Blackwell 6000 Pro | S-forum-gemma4-26b-bench |

> **[conjecture]** **Qwen 122B vLLM v26 + fp8 KV + DFlash + int8 lm-head on single Spark**
> (S-forum-qwen122-v26-dflash, styles01): the first working fp8 KV + DFlash implementation on
> GB10 for a hybrid quantization model. Three custom patches (inc_hybrid, int8_lmhead_v3,
> prefix_align) on vLLM v26 main (commit 318b527). KV cache 549K→1.37M tokens (2.6×), concurrency
> 2.09×→5.24× at 256K context. Decode 45.98 tok/s (recovered from 43.6 via int8 lm-head), prefill
> 957 tok/s (+32% over bf16 KV baseline). The int8 lm-head technique (~1.4 GB → ~175 MB) is a
> GB10-specific memory-reclamation approach. See `[[wiki/models/qwen.md]]` for full details.
>
> **[conjecture]** **Official llama.cpp Docker image on GB10** (S-forum-llamacpp-fastest):
> `ghcr.io/ggml-org/llama.cpp:full-cuda13` matches custom builds at 72.28 tok/s (Qwen3.5-35B-A3B
> NVFP4 GGUF). `--mmap 0` mandatory on UMA. Performance degradation (40→67 tok/s) fixed by
> system update + power-cycle — corroborates the power-controller wedge pattern. See
> `[[wiki/containers-and-tooling.md]]` for full details.

## Forum-reported benchmarks (2026-07-29 ingest, Batch 41)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| Qwen3.6-35B-A3B (Unsloth-Fast) | NVFP4 + flashinfer_b12x | vLLM 0.25.0 | 1 | 435.84 (agg @100 conc) / ~4.4 (per-req) | 1000 in/out | TPOT 212.83 ms; ITL 210.97 ms; Unsloth+b12x ~8% faster than nvidia+Marlin; single Spark | S-forum-unsloth-b12x ||
||| Qwen3.6-35B-A3B (nvidia) | NVFP4 + Marlin | vLLM 0.25.0 | 1 | 404.24 (agg @100 conc) / ~4.0 (per-req) | 1000 in/out | TPOT ~228 ms (est); default Marlin backend; single Spark | S-forum-unsloth-b12x ||
||| Qwen3-4B | NVFP4 KV | SGLang (dev-cu13) | 1 | — | — | KV pool 2,309,504 tokens (1.68× FP8); dtype torch.float4_e2m1fn_x2; flashinfer prefill + trtllm_mha decode | S-forum-nvfp4-kv ||
||| Qwen3-4B | FP8 KV (fp8_e4m3) | SGLang (dev-cu13) | 1 | — | — | KV pool 1,371,456 tokens; baseline for NVFP4 KV comparison | S-forum-nvfp4-kv ||
||| Qwen3-4B | NVFP4 KV | SGLang (dev-cu13) | 1 (6000 Pro) | — | — | KV pool 1,808,192 tokens (1.69× FP8); TPOT 17.04 ms; 5,275 tok/s output @100 conc; RTX PRO 6000 Blackwell | S-forum-nvfp4-kv ||
||| Qwen3-4B | FP8 KV (fp8_e4m3) | SGLang (dev-cu13) | 1 (6000 Pro) | — | — | KV pool 1,067,328 tokens; RTX PRO 6000 Blackwell baseline | S-forum-nvfp4-kv ||
||| DeepSeek-V4-Flash (REAP25) | IQ2_XXS+MXFP4+MXFP8 mix + DSpark | ds4-server (twaggs88 fork) | 1 | 16.5 (spec, 0-8k) / ~24 (structured, v0.2.3) | 1M | 92/100 tool-eval; 77.2% DSpark acceptance; 91 GB resident; 420→390 pp tok/s; 410-430 pp (v0.2.3 W4A8) | S-forum-dsv4-reap25 ||
||| DeepSeek-V4-Flash (REAP25) | IQ2_XXS+MXFP4+MXFP8 mix + DSpark | ds4 (marco.palaferri fork) | 1 | 24-25 (DSpark, 55k-70k ctx) | 55k+ | 854 pp tok/s (first 8k chunk); 787 pp (13.6k prompt); 724 pp (41.7k append @55k ctx); HMMA attention | S-forum-dsv4-reap25 ||

> **[conjecture]** **Unsloth+b12x vs nvidia+Marlin on Spark** (S-forum-unsloth-b12x, shahizat):
> Unsloth aggregate 435.84 vs nvidia 404.24 tok/s at 100 concurrent — a ~8% Unsloth lead. This
> reverses the prior [reported] finding (Unsloth ~15% slower) — the difference is the b12x
> backend (Unsloth uses b12x, nvidia uses Marlin). A controlled comparison isolating the backend
> variable is needed. Single source → [conjecture]. See `[[wiki/models/qwen.md]]`.
>
> **[conjecture]** **NVFP4 KV cache 1.68× capacity over FP8 on Spark** (S-forum-nvfp4-kv, shahizat):
> Qwen3-4B on SGLang: NVFP4 KV pool 2.31M tokens vs FP8 1.37M tokens. The dtype is
> `torch.float4_e2m1fn_x2`. Quality validation recommended before production use. Single source.
>
> **[conjecture]** **DSV4-Flash REAP25 on single GB10 — 16.5 tok/s spec decode, 92/100 tool-eval**
> (S-forum-dsv4-reap25, twaggs88): a third independent ds4 fork with measured-KL quant allocation.
> The IQ2+MXFP4+MXFP8 mixed format beats hand-picked allocation by 8 composite points. marco.palaferri's
> fork achieves 854 tok/s prefill via HMMA attention. DSV4-Flash prefill is compute-bound (tensor
> cores), not bandwidth-bound — distinct from decode. See `[[wiki/engines.md]]` for full details.

## Image generation benchmarks (diffusion models on GB10, 2026-07-10)

**[conjecture]** — single-source forum benchmarks for diffusion models on DGX Spark, all 1024×1024,
single-node, via `diffusers` library (not ComfyUI). Generation time post-compile (torch compile needs
~2 warmup runs). See `[[wiki/quantization-on-gb10.md]]` for NVFP4 W4A4 FLUX.2 details.

||| Model | Steps | Time (BF16) | Time (NVFP4) | Notes | Source ||
|||---|---|---|---|---|---|||
||| FLUX.2-klein-9B | 4 | 4.4s | 3.3s | distilled; NVFP4 via torchao W4A4 | S-forum-diffusion-speeds ||
||| Z-Image-Turbo | 9 | 7.2s | 5.6s | BF16 mostly; torch compile 8.1s (default attn) | S-forum-diffusion-speeds ||
||| ERNIE-Image-Turbo | 8 | 8.8s (BF16) / 11.2s (orig) | 6.4s | `DIFFUSERS_ATTN_BACKEND=_native_cudnn` improved 11.2→8.8 | S-forum-diffusion-speeds ||
||| SDXL 1.0 | 30 | 11.3s | — | BF16; no NVFP4 tested | S-forum-diffusion-speeds ||
||| Krea2-Turbo | 8 | 13.9s (BF16+optimized) | 12.4s | Default 39.3s; `torch.set_float32_matmul_precision('high')` → 32s; `DIFFUSERS_ATTN_BACKEND=_native_cudnn` → 13.9s | S-forum-diffusion-speeds ||
||| Qwen-Image-2512 | 50 | 61.0s | — | ~50% memory used; no NVFP4 tested yet | S-forum-diffusion-speeds ||

- **[conjecture]** **`DIFFUSERS_ATTN_BACKEND=_native_cudnn`** is a significant GB10 diffusion speedup
  env var (S-forum-diffusion-speeds, ijontichy): improved Krea2-Turbo 39.3→13.9s, ERNIE-Image-Turbo
  11.2→8.8s, no effect on Z-Image-Turbo. Combined with `torch.set_float32_matmul_precision('high')`.
  Second source (vasimv) reports 15-17s for Krea2-Turbo FP16 on ComfyUI with 610 drivers + CUDA 13.3.
- **[conjecture]** **NVFP4 W4A4 quantization gives modest speedups on diffusion models** (S-forum-diffusion-speeds):
  FLUX.2-klein 4.4→3.3s (~1.3×), Z-Image-Turbo 7.2→5.6s (~1.3×), ERNIE 8.8→6.4s (~1.4×), Krea2 13.9→12.4s
  (~1.1×, marginal). Smaller gains than the ~3× seen on FLUX.2-dev (which uses activation-quantized
  real FP4 compute via Triton — see `[[wiki/quantization-on-gb10.md]]`). The difference: these are
  weight-only NVFP4 vs FLUX.2-dev's activation-quantized path.

## Acer Veriton GN100 thermal A/B test (2026-07-30 ingest)

**[conjecture]** — single-source forum benchmark, 2 × Acer Veriton GN100 (DGX Spark OEM),
Qwen3.5-122B-A10B INT4 AutoRound + DFlash, vLLM (`aeon-vllm-ultimate`), 1h continuous
`llama-benchy` (pp2048/tg512, concurrency 3, 300 runs, no cache). S-forum-acer-thermal.

| Unit | Idle temp | Load temp (sustained) | Peak spike | GPU util | tok/s/req | Errors |
|---|---|---|---|---|---|---|
| A | 42°C | 68-74°C | 82°C (brief, recovered) | 96% | ~25 | 0 |
| B | 43°C | 68-70°C | 70°C | 96% | ~25 | 0 |

- Both units: zero thermal throttling, zero throughput degradation over 1 hour, 447/435
  requests completed. CPU usage low (6.3% / 5.3% avg). GPU-bound decode workload.
- Acer chassis peaks ~68°C vs 80-82°C for other OEM builds (per StorageReview comparison).
- After the test, both units idled at ~40°C. Mini-rack, no extra fans.
- **Durable finding:** the Acer Veriton GN100 runs ~12-14°C cooler than other OEM SKUs
  under identical sustained inference load — first published Acer thermal data point.

## Batch 45 forum ingest (2026-07-31)

**[conjecture]** — all single-source forum benchmarks unless noted. S-forum-sm121-4bugs,
S-forum-velogb10, S-forum-hy3-1bit, S-forum-laguna-king.

| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |
|---|---|---|---|---|---|---|---|
| gpt-oss-120B | MXFP4 | vLLM 0.17.1 patched | 1 | **59** [reported] | 131K | 4-bug fix + 6 gpt-oss pitfalls; `--enforce-eager` costs 26→59; corroboration: raphael.amorim 58-60 tok/s | S-forum-sm121-4bugs |
| Qwen3.5-35B (BF16→MXFP4 online) | MXFP4 | vLLM 0.17.1 patched | 1 | **59** | 200K | same 4-bug fix; raphael.amorim: FP8 52-55 tok/s | S-forum-sm121-4bugs |
| Qwen3.5-122B | NVFP4 Marlin W4A16 | vLLM 0.17.1 patched | 1 | ~15 | 200K | raphael.amorim: int4-AutoRound 28-29 tok/s | S-forum-sm121-4bugs |
| Qwen3.6-27B (dense) | NVFP4-full (100%) | veloGB10 | 1 | ~40 | — | pure NVFP4, all layers quantized | S-forum-velogb10 |
| Qwen3.6-27B (dense) | NVFP4-full (100%) | veloGB10 | 2 | ~45-50 | — | TP=2; community: slower than vLLM for 27B dense | S-forum-velogb10 |
| Qwen3.6-35B-A3B (MoE) | NVFP4-full (100%) | veloGB10 | 1 | ~110 | — | pure NVFP4; at parity with eugr vLLM at c=1 | S-forum-velogb10 |
| Qwen3.6-35B-A3B (MoE) | NVFP4-full (100%) | veloGB10 | 2 | ~120+ | — | TP=2; vLLM wins at c=4/8/16 | S-forum-velogb10 |
| Qwen3.6-9B (dense) | NVFP4-full (100%) | veloGB10 | 1 | ~80 | — | pure NVFP4 | S-forum-velogb10 |
| Hy3-295B | 1-bit GGUF | llama.cpp | 1 | ~15 | — | tight fit even at 1-bit; very intelligent but slow | S-forum-hy3-1bit |
| Laguna-S-2.1 | NVFP4 (updated) | vLLM 0.25.1 | 1 | 19-50 | 256K | wide range: Schampuswerner 19-24, nuk3s 22.6, vr8vr8 40-50, robert287 45.5 code/27.2 structured | S-forum-laguna-king |

## Batch 47 forum ingest (2026-08-02)

**[conjecture]** — all single-source forum benchmarks. S-forum-nemotron-2node,
S-forum-dsv4-dspark-eugr.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Nemotron-3-Super-120B-A12B | NVFP4 | vLLM TP=2 (Ray) | 2 | 13.67–14.33 | 262K | dual-node slightly slower than single-node (15 tok/s); TRITON_ATTN, cutlass MoE, fp8 KV, mamba_ssm_cache_dtype float32, fastsafetensors | S-forum-nemotron-2node ||
|| DeepSeek-V4-Flash-DSpark | NVFP4 (`nvfp4_ds_mla` KV) | vLLM+DSpark spec=3 TP=2 (Ray) | 2 | 71.63 (c50 output) | 262K | 3 draft tokens beats 5; 48.35% acceptance, 52.52 ms TPOT; max_num_batched_tokens 10240; FlashInfer PR 3817 | S-forum-dsv4-dspark-eugr ||

## Batch 50 forum ingest (2026-08-03)

**[conjecture]** — all single-source forum benchmarks. S-forum-4node-qrs812,
S-forum-laguna-yaml.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| DeepSeek-V4-Flash-0731 | NVFP4 (`nvfp4_ds_mla` KV) + DSpark spec=3 | vLLM 0.21.1rc1.dev339 TP=4 | 4 (QRS812) | ~90 (C=1) / ~40.4 (C=6 per-req) | 512K | cold prefill ~2500 tok/s; KV cache hit effective prefill ~193K tok/s; QRS812 switch fabric; mashie challenges: C12 2-node=230 vs 4-node=209 | S-forum-4node-qrs812 ||
|| Laguna-S-2.1-NVFP4 | NVFP4 W4A4 + DFlash spec=15 | vLLM (eugr spark-vllm-docker) TP=2 | 2 | 122.63 (aggregate output, c50) | 262K | 268.58 tok/s total throughput; DFlash acceptance 11.71%, accept_len 2.76; per-pos: pos0=64.89% → pos14=0.78%; TP=1 option for single Spark; --kv-cache-memory=32449423258 override; model is retired (see models/laguna-s-2.1.md) | S-forum-laguna-yaml ||

## Batch 51 forum ingest (2026-08-04) — ComfyUI diffusion benchmarks

**[conjecture]** — single-source forum benchmarks via ComfyUI (not diffusers). S-forum-comfyui-triplany.

|||| Model | Steps | Time (BF16/fp8) | Time (NVFP4/full) | Mem | Notes | Source |||
||||---|---|---|---|---|---|---|||
|||| Z-Image-Turbo t2i | (template) | 96.17s cold / 43.73s warm | — | 43.5 GB | bf16; stock ComfyUI template; single Spark | S-forum-comfyui-triplany |||
|||| Flux2-dev t2i | (template) | 300.38s cold / 50.14s warm | — | 68 GB | fp8mixed; stock template | S-forum-comfyui-triplany |||
|||| Flux2-dev (full) + mistral3_small | (template) | — | 407.52s cold / 80.25s warm | 93.80 GB | full quant + bf16 text encoder; highest memory workflow | S-forum-comfyui-triplany |||
|||| LTX 2.3 t2v | (template) | 179.55s cold / 81.83s warm | — | 44.73 GB | fp8; 1280×720, 5s duration | S-forum-comfyui-triplany |||
|||| LTX 2.3 22B | (video) | — | ~12 min | — | NVFP4; 20s video; quality "not too bad" | S-forum-comfyui-triplany |||
|||| Wan2.2 14b t2i | (template) | 644.75s cold / 565.24s warm | — | 18 GB | fp8; 640², 5s duration | S-forum-comfyui-triplany |||
|||| Flux1-dev (full) + t5xxl | (template) | — | 113.17s cold / 32.61s warm | 32.16 GB | full quant + fp16 text encoder | S-forum-comfyui-triplany |||

> **[conjecture]** **ComfyUI benchmarks on DGX Spark** (S-forum-comfyui-triplany, Triplany):
> cold-vs-warm timing across 6 diffusion workflows on a single Spark, via a patched ComfyUI
> setup targeting UMA memory management. The warm/cold ratio ranges from ~2.2× (Z-Image) to
> ~6× (Flux2-dev fp8mixed) — the first warm run benefits from models staying resident in UMA.
> Flux2-dev full quant + mistral3_small peaks at 93.80 GB (near the 121 GB ceiling). LTX 2.3
> 22B NVFP4 is the first reported NVFP4 video model data point on GB10. See
> `[[wiki/containers-and-tooling.md]]` → Batch 51 for the full setup details.

## Batch 52 forum ingest (2026-08-04)

**[conjecture]** — all single-source forum benchmarks. S-forum-dsv4-0731-bench.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| DeepSeek-V4-Flash-0731 | NVFP4 (`nvfp4_ds_mla` KV) + MTP | vLLM 0.25.2.dev0 TP=2 | 2 | 35.3 (B1 e2e) / 65.6–69.5 (C4) | 524K | TP=2 reference config; 40.1% MTP acceptance; KV pool 345K tok | S-forum-dsv4-0731-bench ||
|| DeepSeek-V4-Flash-0731 | NVFP4 + MTP | vLLM TP=4 seqs=32 | 4 | 46.8–48.6 (B1, +33%) / ~101 (C4) / 333–344 (C32) | 524K | best config; 39.8–40% acceptance; KV pool 1.93–1.98M (7.81× vs TP=2) | S-forum-dsv4-0731-bench ||
|| DeepSeek-V4-Flash-0731 | NVFP4 + MTP | vLLM DP4EP | 4 | 31.3 (B1) / 75.7–95.0 (C4) / ~233 (C32) | 524K | data parallel ×4 expert parallel; 40–44% acceptance; KV pool 1.59M ×4 | S-forum-dsv4-0731-bench ||
||| DeepSeek-V4-Flash-0731 | NVFP4 | vLLM TP2PP2 (Ray, no spec) | 4 | 22.8 (B1) / ~56 (C4) / ~83 (C16 sat) | 524K | pipeline parallel; saturates at C16; KV pool 4.09M | S-forum-dsv4-0731-bench ||

## Batch 53 forum ingest (2026-08-05)

**[conjecture]** — single-source forum benchmarks. S-forum-jul31-wedge.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
||| Qwen3.6-35B-A3B | NVFP4 + MTP | vLLM v0.25.0 | 1 | 107 (pre-wedge) → 45 (wedged) → 84 (post-fix) | — | Power-controller wedge triggered by July 31 apt upgrade; MTP acceptance 79.81% → 50.02% post-fix; AC power-cycle fix | S-forum-jul31-wedge ||

## Batch 54 forum ingest (2026-08-05)

**[conjecture]** — all single-source forum benchmarks. S-forum-dsv4-0731-dspark-loader,
S-forum-macaron-v1-tall.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| DeepSeek-V4-Flash-0731-DSpark | NVFP4 (`nvfp4_ds_mla` KV) + DSpark k=5 | vLLM TP=2 (Ray) | 2 | 55.4 mean / 66.1 peak (post-fix) / 32.7 (pre-fix) | 1M | DSpark draft loader fix: shared_experts.w1/w3 → gate_up_proj mapping; acceptance 25.7%→60.2%; SSE stream:false required for accurate tok/s | S-forum-dsv4-0731-dspark-loader ||
||| DeepSeek-V4-Flash-0731-DSpark | NVFP4 + DSpark k=5 | vLLM TP=2 (Ray) | 2 | 12.03 (srivatsa1, pre-fix) | — | Draft quant-config inheritance bug: draft inherits target NVFP4 config → ModelOptNvFp4FusedMoE on FP8 draft weights → 1.5-4.5% acceptance; fix: strip target-only keys from draft quant_config (vLLM PR #49133) | S-forum-dsv4-0731-dspark-loader ||
||| Macaron-V1-Tall (50B) | bf16 + fp8 KV | vLLM (spark-vllm-docker vllm-node) | 1 | 25-27 (no MTP) / 41.93-42.79 (MTP nst=3, +2%) | 229376 | 35B Qwen3.6-35B-A3B base + 4× 3.7B LoRA specialists; TP=1, util 0.7; MTP 71.5% acceptance but only +2% throughput; tool-eval base 90/100, full router 82/100 | S-forum-macaron-v1-tall ||

> **[conjecture]** **DSV4-Flash-0731-DSpark with loader fix: 55.4 tok/s mean, 66.1 peak**
> (S-forum-dsv4-0731-dspark-loader, tonyd615): the DSpark draft loader weight-mapping fix
> (shared_experts.w1/w3 → gate_up_proj) restores acceptance from 25.7% to 60.2% and
> throughput from 32.7 to 55.4 tok/s (+69%) on 2× Spark TP=2 k=5 NVFP4 KV 1M context.
> This is the highest reported DSV4-Flash-0731 throughput on 2× Spark. A second user
> (srivatsa1) hit a different draft-quant-config-inheritance bug (vLLM PR #49133) that
> collapsed acceptance to 1.5-4.5% (12 tok/s) — fixed by stripping target-only ModelOpt
> keys from the draft's quantization_config. Both are vLLM config-plumbing bugs, not
> GB10 hardware issues, but they bite every Spark user running DSV4-Flash-DSpark.
>
> **[conjecture]** **Macaron-V1-Tall at 25-27 tok/s bf16** (S-forum-macaron-v1-tall,
> TheAwakenOne): ~half the speed of NVFP4 Qwen3.6-35B-A3B because it runs bf16 (no quant).
> MTP nst=3 adds only +2% throughput despite 71.5% acceptance — consistent with the
> proven finding that MTP on this model family can be marginal. Tool-eval: the Macaron
> Tool-eval: base Qwen 90/100, full Macaron router 82/100 because
> routing sends most requests to L0 general chat instead of the tool specialist.

## Batch 55 forum ingest (2026-08-06) — MiniMax-H3 video generation

**[conjecture]** — single-source forum benchmarks via ComfyUI on single Spark. S-forum-minimax-h3-comfyui.

||||| Model | Workflow | Resolution / Duration | Time | Notes | Source |||
|||---|---|---|---|---|---|||
||| MiniMax-H3 | i2v (image-to-video) | 0.2M, 5s | 174s | single Spark, ComfyUI; models from Comfy-Org/MiniMax-H3 | S-forum-minimax-h3-comfyui |||
||| MiniMax-H3 | t2v (text-to-video) | 0.2M, 5s | 143s | single Spark, ComfyUI | S-forum-minimax-h3-comfyui |||
||| MiniMax-H3 | r2v (reference-to-video) | 0.2M, 5s, 2 ref imgs | 215s | single Spark, ComfyUI | S-forum-minimax-h3-comfyui |||
||| MiniMax-H3 | i2v (768²) | 768×768, 5s | ~235s | with easycache + SageAttention KJ nodes | S-forum-minimax-h3-comfyui |||
||| MiniMax-H3 | i2v (768²) | 768×768, 10s | 432s | with easycache + SageAttention KJ nodes | S-forum-minimax-h3-comfyui |||

> **[conjecture]** **MiniMax-H3 video generation on DGX Spark** (S-forum-minimax-h3-comfyui,
> wxhpad + cx77 + TheAwakenOne): first reported MiniMax-H3 video diffusion data points on
> GB10. Generation times 143–215s for 5s/0.2M video depending on workflow type (t2v fastest,
> r2v slowest). At 768² resolution, ~235s for 5s video, 432s for 10s — roughly linear with
> duration. easycache (native ComfyUI) + KJNodes SageAttention nodes used for speedup.
> Consistent with the broader ComfyUI-on-GB10 pattern (compute-bound video diffusion, UMA
> memory management). See `[[wiki/containers-and-tooling.md]]` → Batch 55.

## Batch 56 forum ingest (2026-08-06)

**[conjecture]** — all single-source forum benchmarks. S-forum-dsv4-0731-ds4-cuda,
S-forum-laguna-modelopt, S-forum-vllm-qemu.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| DeepSeek-V4-Flash-0731 | IQ2XXS + Q2K KV + Q8 attn/shared | ds4 CUDA (Entrpi/ds4 fork v0.5.4) + DSpark k=2 | 1 | ~40 | 131K | native C/CUDA binary; DSpark MTP k=2; 1M ctx fits ~107GB with kv-disk-dir offload (coder543) | S-forum-dsv4-0731-ds4-cuda ||
|| Laguna-S-2.1 | ModelOpt NVFP4 W4A4 | vLLM | 1 | 28 | — | 88/100 agent tool calls; JasonW2025/Laguna-S-2.1-ModelOpt-NVFP4-W4A4-vllm; model is retired (see models/laguna-s-2.1.md) | S-forum-laguna-modelopt ||
|| Qwen2.5-Coder-32B-Instruct | (via x86_64 Docker → QEMU) | vLLM (x86_64 image) | 1 | 3.7 | — | QEMU emulation trap — x86_64 Docker image on Grace ARM64 CPU; baseline for "how slow QEMU is" vs native ARM64 | S-forum-vllm-qemu ||

> **[conjecture]** **DSV4-Flash-0731 on ds4 CUDA engine — 40 tok/s single Spark** (S-forum-dsv4-0731-ds4-cuda,
> styles01): the ds4 custom CUDA engine (Entrpi/ds4 fork v0.5.4) achieves 40 tok/s on a single
> Spark with IQ2XXS quant + DSpark MTP k=2 at 131K context — a ~43% improvement over the original
> ds4 Q2 baseline (~28 tok/s, S-forum-ds4-cuda). coder543 reports 1M context fits in ~107 GB with
> `DS4_CUDA_NO_HBM_CACHE=1` + `--kv-disk-dir` for KV cache offload. See `[[wiki/engines.md]]` →
> Batch 56 for the full recipe and env vars.
>
> **[conjecture]** **Laguna-S-2.1 ModelOpt NVFP4 W4A4 — 28 tok/s, 88/100 tool calls** (S-forum-laguna-modelopt,
> JW2026): a new ModelOpt W4A4 quant variant of the retired Laguna-S-2.1 model. The 28 tok/s figure
> is consistent with the existing Laguna-S-2.1 range (19-50 tok/s depending on quant/config, see
> models/laguna-s-2.1.md). The model is retired — this data point is recorded for completeness only.
>
> **[conjecture]** **QEMU emulation baseline: 3.7 tok/s** (S-forum-vllm-qemu, rithinsundar87):
> Qwen2.5-Coder-32B-Instruct via x86_64 vLLM Docker image on Grace CPU = 3.7 tok/s. This is the
> "wrong architecture" baseline — native ARM64 images should be 5-10×+ faster. See
> `[[wiki/platform-gb10.md]]` → Batch 56 for the full QEMU emulation trap finding.

## Batch 58 forum ingest (2026-08-07)

**[conjecture]** — all single-source forum benchmarks. S-forum-sparkring,
S-forum-dsv4-llamacpp-fan.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| GLM-5.2 | MXFP4-Experts-GPTQ | SparkRing SIRCL TP4/DCP4/MTP4 | 4 (switchless ring) | 19-20 (C1) / 50-63 (C8 agg) | 500K | `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`; nvfp4_ds_mla KV + per-token scaling; 30s sustained decode; prefill 796-876 tok/s; C8 peak 66.3 (workload-dependent) | S-forum-sparkring ||
|| GLM-5.2 | MXFP8-NVFP4-NF3 hybrid | SparkRing SIRCL TP4/DCP4/AMTP2-4 | 4 (switchless ring) | 40-50 (C4 shared ctx) | 875K | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`; 64 NVFP4 + 192 NF3 experts; NVFP4 MLA + FP8 RoPE KV | S-forum-sparkring ||
|| GLM-5.2 | MXFP4-Experts-GPTQ | SparkRing SIRCL TP4/DCP4 (eager) | 4 (switchless ring) | 18.3 (median) | 13K | Terry01 independent reproduction; eager mode (CUDA graphs produce single-token lock); ~92% of published eager number | S-forum-sparkring ||
|| DeepSeek-V4-Flash-0731 | UD-IQ2_M | llama.cpp (llama-server) | 1 (HP ZGX) | 16.2 (tg32) | 524K | `--flash-attn on --ctx-size 524288 --parallel 4 --no-mmap --threads 10`; pp2048 390 tok/s; ttfr 4860ms; tg32 degrades 16.2→15.26 at 16K depth; firmware update fixed thermal shutdown (71°C/75W) | S-forum-dsv4-llamacpp-fan ||

> **[conjecture]** **GLM-5.2 on SparkRing SIRCL — 19-20 tok/s C1, 50-63 tok/s C8 aggregate**
> (S-forum-sparkring, FujitsuPolycom): the first reported custom-RDMA-collective inference stack
> on GB10. The C1 decode (19-20 tok/s) is consistent with the 20-25 tok/s range across other GLM-5.2
> 4× Spark recipes (AWQ-INT4, NVFP4, Hybrid FP8+MXFP4) — the quant format matters less than the
> sparse-MLA attention + bandwidth-bound decode ceiling. The C8 aggregate (50-63 tok/s) shows
> good concurrency scaling under shared-prefix workloads. See `[[wiki/models/glm-5.2.md]]` →
> SparkRing section.
>
> **[conjecture]** **DeepSeek-V4-Flash-0731 UD-IQ2_M via llama.cpp on HP ZGX — 16.2 tok/s tg32**
> (S-forum-dsv4-llamacpp-fan, chrm): single-node llama.cpp serving on HP ZGX (GB10 variant).
> The 16.2 tok/s at tg32 with IQ2_M (2-bit UD quant) and 524K context is consistent with the
> bandwidth-bound decode ceiling for a ~440B model at 2-bit on a single 121 GB node. Prefill
> 390 tok/s (pp2048) is low — llama.cpp's CPU-side processing on Grace limits prefill vs vLLM's
> CUDA prefill path. The `--no-mmap` flag is consistent with the proven UMA requirement
> (`[[wiki/llama-cpp-rpc.md]]`). The firmware update improving thermals (71°C/75W, no shutdown)
> corroborates the documented EC firmware / fan curve findings (`[[wiki/platform-gb10.md]]`).

## Batch 59 forum ingest (2026-08-08)

**[conjecture]** — single-source forum benchmark. S-forum-kimi-k3-coder-reap.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| Kimi K3 Coder REAP-320 | MXFP4 | llama.cpp (llama-bench) | 8 (TP=8) | 23.79 (d0) / 29.69 (d4000) / 21.29 (d32000) | 32K | pp2048 541-686 tok/s; peak 35 tok/s; REAP variant "loops a lot" (quality issue); full K3 needs 16× GB10; same active experts as full model | S-forum-kimi-k3-coder-reap ||

> **[conjecture]** **Kimi K3 Coder REAP-320 MXFP4 on 8× GB10** (S-forum-kimi-k3-coder-reap,
> ciprianveg): first reported Kimi K3 variant on DGX Spark. Decode 21-30 tok/s (tg1500), peak
> 35 tok/s, prefill 541-686 tok/s. Decode is relatively flat across context depths (d0-d32000),
> suggesting attention is not the bottleneck at these depths. The REAP pruned variant produces
> repetitive output ("loops a lot") — the OP does not recommend it. Consistent with the
> bandwidth-bound decode regime for large MoE at MXFP4 on 8× Spark. See
> `[[wiki/models/kimi-k3.md]]`.
