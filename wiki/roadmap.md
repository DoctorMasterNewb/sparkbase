# Roadmap — open problems & areas of further development

> **area:** roadmap
> **status:** open-problem
> **evidence:** mixed
> **sources:** S-xnode-cudagraph, S-m3-20tps, S-sess-jun4, S-sess-jun5, S-mimo-results, S-dgxspark-report, S-forum-mxfp4-patches, S-forum-cx7-13gbps, S-forum-nvfp4-100b, S-forum-cx7-bricked, S-forum-sdpa-corruption, S-forum-nvmeof-expert, S-forum-vllm-019-vs-023, S-forum-colibri-glm52, S-forum-glm52-8x, S-forum-bonsai27b, S-forum-mtp-lossless, S-forum-ec-fan-rollback, S-forum-ec-fan-asus, S-forum-inkling
> **updated:** 2026-07-19

The unsolved stuff. Each item links to the page with the detail. Close an item by moving its finding
onto the relevant page and deleting it here.

## Platform / kernels (the big walls)

- **Cross-node cudagraph capture for MiniMax-M3** — M3's fused host-staged all-reduce can't be captured
  by breakable_cudagraph; filed vllm#46253. **[reported]** **Upstream PR #46372** now implements our
  eager-break direction (narrowed to all-reduce after our review) — fixes the capture *crash* but
  **[conjecture]** likely not yet the *garbage replay* (in-place copy-back ≠ static buffers). A staged
  GB10 test (minimal 1-op repro + patched dev537 image) is ready; next action = run it on go-ahead and
  report to the PR. Unlocking replay is the biggest throughput win for M3-class cross-node MoE.
  `[[wiki/cudagraphs-and-compile.md]]`
- **[proven]** **MoE cudagraph capture on sm_121 (single-node)** — large-expert MoE crashes capture;
  eager-only. Hardware/arch level. `[[wiki/cudagraphs-and-compile.md]]`
- **[proven]** **Native FP4/FP8-block-scale compute** — absent on GB10; all FP4 + block-scale FP8 run
  Marlin weight-only decompress (compute-bound at high batch, caps aggregate ~900 tok/s). Inherent to
  Spark; **[conjecture]** would scale on datacenter Blackwell. `[[wiki/quantization-on-gb10.md]]`

## Speeding up specific regimes

- **DeepSeek-V4-Flash-DSpark — the cross-node MoE that DOES work at speed (2026-06-30).** Where M3
  deadlocks, this serves: **vLLM** (custom DSpark overlay,
  `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` base → 4-stage NVFP4 build) + **self-speculative
  decode** (`--speculative-config '{"method":"dspark","num_speculative_tokens":5}'`) cross-node TP=2
  `mp`. **[proven]** The spec-decode amortizes the host-staged all-reduce: ~3.5–4.0 accepted tokens per
  forward step. **[reported]** (DSpark mechanism, per S-dgxspark-report: a heavy *parallel backbone*
  drafts a whole block cheaply, then a lightweight rank-256 *Markov head* applies a prefix-dependent
  bias so acceptance holds deep into a 5-token block; *confidence-scheduled verification* truncates
  block length as concurrency rises — the reported +60–85% over single-token.) **[proven] Measured: ~35
  tok/s warm single-stream (code), 1M context, 1.91M-token NVFP4 KV pool (`nvfp4_ds_mla`), 1.78×
  concurrency @ 1M**; weights 79.5 GiB/node (true NVFP4). **[reported]** Forum 374846 claims 56.73
  (p512/g256) — we hit ~62% of that; our spec acceptance is actually *higher*, so the gap is our slower
  cross-node **forward step** (host-staged, no GPUDirect). Why it works where M3 didn't: vLLM cross-node
  mp is solid here (mimo proves it) + spec-decode hides the collective latency. Deploy via a compose
  stack (needed our `VLLM_HOST_IP=fabric` per-node patch — see
  `[[wiki/multinode-tp-and-networking.md]]`). **[proven] The ~35-vs-56 gap was NOT our config** — the
  head GPU was wedged in the DGX Spark "14 W cap" power-controller bug (pinned 611 MHz/12 W under load
  while the worker ran 2431 MHz/35 W), dragging the lockstep TP=2 pair. Fix = AC power-cycle the head
  (`[[wiki/platform-gb10.md]]` → power-controller wedge); re-benchmark after. Chat endpoint needs
  `thinking:false` for plain `content`.

- **[proven] Fast MiniMax-M3 — SOLVED 2026-07-03 via EAGLE3 spec decode: 13.7 tok/s prose / 15 code
  (peak 20), vision working, deployed as swapper stack `minimax-m3-eagle3`.** The DSpark mechanism (spec
  decode amortizes the host-staged all-reduces) transplanted via the `Inferact/MiniMax-M3-EAGLE3` draft
  on dev537. Full detail + tuning A/B (nst=5/draft_tp=1 is a measured regression) in
  `[[wiki/models/minimax.md]]`. Historical dead-ends kept there too: SGLang forward-deadlock
  (2026-06-30), REAP50 = 120 GiB (doesn't fit one node), PP2+EAGLE3 impossible (draft lacks
  SupportsPP), PP2 byte-uneven layer split OOM'd the worker (use `VLLM_PP_LAYER_PARTITION` if ever
  retried). Remaining upside for M3-class: cudagraph replay (below) or faster interconnect.
- **[proven]** **MTP that actually pays off cross-node** — MTP ≈ 0 gain without cudagraphs, so it's
  stuck behind the cudagraph wall on Spark. `[[wiki/models/step-3.7.md]]`
- **Capping idle KV without killing throughput (Atlas)** — **[proven]** no clean lever
  (`gpu_memory_utilization`/`oom_guard_mb`/`block_size` don't bind; high-speed-swap halves throughput).
  **[conjecture]** Untested promising route: a docker `--memory` cgroup limit via a custom launch
  wrapper (sparkrun has no docker-arg passthrough). `[[wiki/engines.md]]`

## Model bring-ups still open

- **gemma-4-12B unified (audio-tower) on Atlas** — does Atlas's gemma4 loader accept `gemma4_unified`
  vs the text+vision arch? Bring-up was cut off. `[[wiki/models/gemma-4.md]]`
- **Uncensored Qwopus3.6 on Atlas** — **[proven]** blocked by inline-MTP + dense-VL loader bugs; open
  paths: AEON-7 NVFP4, self-quantize MoE w/ modelopt, or llama.cpp GGUF. `[[wiki/models/qwen.md]]`
- **A coherent abliterated MiMo-V2.5** — **[proven]** the lovesenko abliteration is damaged; need a
  different uncensored MiMo quant or wait for an author fix. `[[wiki/models/mimo-v2.5.md]]`

## Computer-use (Holo)

- Wire a real computer-use action loop against the Holo endpoint; run Holo's native tool-call smoke
  test; combine no-think + screenshot downscaling to push vision concurrency past the prefill
  bottleneck. `[[wiki/models/holo-3.1.md]]`

## Knowledge-base hygiene

- Mine the remaining session transcripts (and future ones) for benchmark rows and any model not yet
  paged. Re-probe the container table monthly (vLLM arch support moves fast).
  `[[wiki/containers-and-tooling.md]]`

## Forum-sourced open problems (2026-07-08 ingest)

- **[conjecture]** **MXFP4 online quantization upstreaming** — amasawa_seiji's vLLM 0.17.0 patches
  (BF16→MXFP4 online quant for attention + lm_head + MoE, SM121 CUTLASS fixes, GDN kernel fix) give
  +56-65% tok/s on Qwen3.5-35B-A3B and gpt-oss-120b. CUTLASS SFA/SFB fix is a copy-paste bug suitable
  for upstream PR; vLLM patches need refactoring for upstream code quality. Hardware agent could
  reproduce the 70.68 / 80.88 tok/s numbers to promote to `[reproduced]`.
  `[[wiki/quantization-on-gb10.md]]`
- **[conjecture]** **CX-7 PCIe SlotPowerLimit 0W bug** — `lspci` reports `SlotPowerLimit 0W` →
  mlx5_core driver throttles CX-7 to ~13 Gbps (expected ~190 Gbps). Multiple forum users hit this.
  Is this a BIOS/firmware fix, or a driver workaround? Hardware agent should check `lspci -vv`
  SlotPowerLimit on their pair and cross-reference with `ib_write_bw` results.
  `[[wiki/multinode-tp-and-networking.md]]`
- **[conjecture]** **Distributed NVFP4 quantization pipeline** — single-node `modelopt hf_ptq.py`
  OOM-kills on 100B+ models on Spark. The distributed Ray pipeline (layer-sharded, modelopt 0.43)
  has 6 documented bugs (accelerate misdetect, missing input_scale keys, vocab_size=2, calibrator
  lifecycle). Hardware agent with a 2-node setup could reproduce the pipeline and verify the
  Anubis-Pro-105B-NVFP4 output quality. `[[wiki/quantization-on-gb10.md]]`
- **[conjecture]** **GLM-5.2 AWQ-INT4 15% expert prune** — CosmicRaisins reports ~22 tok/s decode on
  4× GB10 with a data-free 15% routed-expert prune (256→218 experts/layer). The prune "might not be
  stable" — hardware agent could test real-world SWE performance vs the full model.
  `[[wiki/benchmarks.md]]`

## Forum-sourced open problems (2026-07-09 ingest)

- **[conjecture]** **CX-7 firmware bricking by auto-updater** (S-forum-cx7-bricked): `mlnx-fw-updater`
  can auto-trigger during routine `apt install` and brick both CX-7 interfaces (stuck in
  `static_config_not_done`, error -110). Hardware agent should: (1) pin/disable the mlnx-fw-updater
  autoupdater, (2) document whether recovery is possible without RMA. `[[wiki/platform-gb10.md]]`
- **[conjecture]** **SDPA EFFICIENT_ATTENTION corruption in community PyTorch builds** (S-forum-sdpa-corruption):
  community-built PyTorch for sm_121 ships with silently broken EFFICIENT attention (output norms
  1.5×–27× off, no NaN). NGC wheels are NOT affected. Hardware agent should: verify which PyTorch
  images produce correct EFFICIENT output, and document the gencode fix for community builds.
  `[[wiki/platform-gb10.md]]`
- **[conjecture]** **NVMe-oF over CX-7 for MoE expert streaming** (S-forum-nvmeof-expert): using the
  second CX-7 QSFP port as an NVMe-oF initiator could enable >128 GB models on a single Spark via
  expert streaming. GB10's unified memory is actually an advantage (no GPUDirect needed — CPU-mediated
  path carries no extra penalty). Internal NVMe gives ~6.6 GB/s (too slow); external NVMe-oF target
  could be faster. **[conjecture]** Colibri (S-forum-colibri-glm52, JustVugg/colibri) now demonstrates
  this approach in practice: a pure-C engine streaming GLM-5.2 (744B MoE) experts from local NVMe
  on a single Spark — O_DIRECT 9.69 GB/s, 2.4-3.3 tok/s. The bottleneck is attention (6.16s of 18s),
  not disk I/O, suggesting faster storage alone won't fix it. Hardware agent with a storage target
  could test NVMe-oF throughput and viability. `[[wiki/multinode-tp-and-networking.md]]`
- **[conjecture]** **vLLM version regression** (S-forum-vllm-019-vs-023): vLLM 0.23 is ~12% slower
  and uses ~15% more memory than 0.19 on Qwen3.5-122B AutoRound. Hardware agent should benchmark
  current vLLM (0.24+) to see if the regression is fixed. `[[wiki/quantization-on-gb10.md]]`

## Forum-sourced open problems (2026-07-17 ingest)

- **[conjecture]** **GLM-5.2-Int4-Int8Mix on 8× GB10 — v16 branch + b12x W4A8 isolated contribution?**
  (S-forum-glm52-8x): the 8× GB10 run reaches ~1,200 t/s prefill / 33–54 t/s decode — a big jump
  over 4× GB10 (~22 tok/s, S-forum-glm52-4x). The OP attributes wins to: (1) v16-unified vLLM branch
  (prefill lever), (2) b12x W4A8 MoE (decode lever), (3) 8× TP scale, (4) DCP1 knobs. But from one
  thread, the individual contribution of each cannot be isolated. Hardware agent with an 8× cluster
  (or even 4× to compare against the existing [reported] 4× baseline) could A/B the v16 branch vs
  base and b12x W4A8 vs Marlin NVFP4. The b12x W4A8 path is particularly interesting — it implies
  INT4 weights + INT8 activations (native FP8 CUTLASS on GB10 for activations), which would be a
  new quant regime not yet characterized on Spark. `[[wiki/quantization-on-gb10.md]]`
- **[conjecture]** **Bonsai 27B 1-bit/ternary kernel path on sm_121** (S-forum-bonsai27b): no GB10
  benchmarks posted. Whether 1-bit/ternary kernels have a working sm_121 path is unverified —
  Marlin does not natively support 1-bit/ternary, so a custom Triton or CUTLASS kernel is needed.
  If it works, the smaller footprint should speed up bandwidth-bound dense decode (the proven
  "fewer bytes = faster" rule). Hardware agent should test Bonsai 27B on a single Spark vs
  Qwen3.6-27B PrismaScout / NVFP4 and report tok/s + quality. `[[wiki/models/qwen.md]]`
- **[conjecture]** **DCP4 decode-aware prefill scheduler — does it help DCP1 too?** (S-forum-glm52-8x):
  the custom decode-aware prefill scheduler (ENABLE_DECODE_AWARE_PREFILL=1) was built for DCP4
  decode starvation but the OP noted it's "very useful also for dcp 1 at long prefill ingestion and
  parallel requests." Hardware agent could test whether the scheduler improves DCP1/TP8 long-context
  concurrent prefill + decode on any model. `[[wiki/multinode-tp-and-networking.md]]`

## Forum-sourced open problems (2026-07-19 ingest)

- **[conjecture]** **Inkling 975B / Inkling-Small 276B MoE on DGX Spark — bring-up not yet
  characterized** (S-forum-inkling): a new multimodal MoE family was announced — **Inkling 975B
  (41B active)** and **Inkling-Small 276B (12B active)**, both with **1M-token context**,
  pretrained on 45T tokens of text/image/audio/video, "native reasoning over text, images, and
  audio." The OP conjectures Inkling-Small 276B in NVFP4 "should run perfectly on Dual Spark,"
  and a second user reports an **8× Spark cluster bring-up is underway** (no recipe, flags, or
  tok/s posted yet). **Why it's on the roadmap, not a model page:** no GB10-specific config,
  quant recipe, benchmark, or error has been reported — only the announcement and intent. Open
  questions for a hardware agent: (1) Does Inkling-Small 276B fit on 2× Spark in NVFP4 (~138 GB
  weights at 4-bit → tight against 242 GB combined but feasible depending on KV/overhead)?
  (2) Does the 1M context fit given GB10's 121 GB/node unified memory and the proven
  decode-bandwidth ceiling? (3) Does the MoE expert count hit the sm_121 cudagraph wall
  (`[[wiki/cudagraphs-and-compile.md]]`)? (4) Which engine (vLLM / SGLang / Atlas / llama.cpp)
  loads the multimodal arch on arm64 first? Promote to a `wiki/models/inkling.md` page once a
  real bring-up with flags + tok/s is reported. Single source (announcement) → [conjecture].
- **[conjecture]** **EC fan-curve regression root cause: EC table vs. SoC/UEFI interaction —
  needs firmware-level isolation** (S-forum-ec-fan-asus, refines S-forum-ec-fan-rollback): the
  ASUS GX10 EC capsule byte-comparison shows the 7-step fan curve is **byte-identical** between
  EC 0x02000004 and 0x02000005 — so the regression is *not* a curve-table edit. The trigger is
  upstream of the curve bytes (SoC/UEFI interaction, an earlier EC version, or an SKU-specific
  difference). Hardware agent with EC telemetry access could: (1) confirm whether the EC is
  actually *following* the documented curve (48%@85°C … 100%@97°C) or ignoring it post-update,
  (2) isolate whether rolling back only the SoC/UEFI (keeping EC) resolves the throttling,
  (3) capture the EC ↔ SoC/UEFI thermal-policy handshake. This would resolve the
  "0x0300xxxx broke the fan profile" attribution and tell ASUS GX10 owners (who have no
  `fwupdmgr downgrade` path) whether a SoC/UEFI-only rollback is viable.
  `[[wiki/platform-gb10.md]]`
- **[conjecture]** **dgx-spark-fieldiag 2.0.4-1 `ofed-scripts` dependency gap — blocks latest
  field diagnostics** (S-forum-ec-fan-asus): `dgx-spark-fieldiag` 2.0.4-1 (latest in the CUDA
  APT repo) depends on `ofed-scripts`, which has no installation candidate in the official
  repositories — so the latest Field Diagnostics cannot be installed. The older 1.0.9-1 works.
  This blocks running current diagnostics on hardware-fault triage (e.g. the thermal regression
  above). Hardware agent should confirm the gap on a fresh image and document whether
  `ofed-scripts` is available from a non-default repo (e.g. MOFED) or must be dropped as a
  dependency. `[[wiki/platform-gb10.md]]`

## Forum-sourced open problems (2026-07-18 ingest)

- **[conjecture]** **MTP quality impact & prefix-cache interaction bug — measure on real Spark**
  (S-forum-mtp-lossless): forum reports MTP measurably affects output quality (up to ~5 pts on
  tool-call bench), and that vLLM + llama.cpp both have MTP+prefix-cache interaction bugs causing
  visible degradation that disappears when prefix caching is off. The thread is split on whether
  practical MTP is "lossy by design" vs "mathematically lossless if implemented correctly." Hardware
  agent should: (1) run MTP-on vs MTP-off on a known model (e.g. Qwen3.6-27B) with prefix caching
  ON and OFF, and measure tool-call-bench / capability-suite deltas; (2) confirm whether the
  quality drift reproduces on a stock vLLM build or only on specific forks; (3) isolate whether
  the prefix-cache interaction is the sole cause. This would promote the quality-impact claim
  from [conjecture] to [reproduced]/[proven] or refute it. `[[wiki/engines.md]]`
