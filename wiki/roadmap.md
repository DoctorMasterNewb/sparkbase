# Roadmap — open problems & areas of further development

> **area:** roadmap
> **status:** open-problem
> **evidence:** mixed
> **sources:** S-xnode-cudagraph, S-m3-20tps, S-sess-jun4, S-sess-jun5, S-mimo-results, S-dgxspark-report, S-forum-mxfp4-patches, S-forum-cx7-13gbps, S-forum-nvfp4-100b, S-forum-cx7-bricked, S-forum-sdpa-corruption, S-forum-nvmeof-expert, S-forum-vllm-019-vs-023, S-forum-colibri-glm52, S-forum-glm52-8x, S-forum-bonsai27b, S-forum-mtp-lossless, S-forum-ec-fan-rollback, S-forum-ec-fan-asus, S-forum-inkling, S-forum-6x-cluster, S-forum-inkling-nvfp4, S-forum-intern-s2, S-forum-pmu-amu, S-forum-6x-ring-rdma, S-forum-gridbook, S-forum-ling3-flash, S-forum-glm52-vision, S-forum-sm121-support, S-forum-inkling-small-2x, S-forum-dsv4-0731-caching, S-forum-powerstress, S-forum-idle-lockup
> **updated:** 2026-08-14

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

- **[conjecture]** **Inkling-Small 276B MoE on 2× DGX Spark — bring-up not yet characterized**
  (S-forum-inkling): the **Inkling 975B (41B active)** bring-up on 8× Spark is now characterized
  (see `[[wiki/models/inkling.md]]`, S-forum-inkling-nvfp4 — NVFP4 clean, long-context decode
  cliff, parked). The **Inkling-Small 276B (12B active)** variant on 2× Spark remains untested.
  Open questions for a hardware agent: (1) Does Inkling-Small 276B fit on 2× Spark in NVFP4
  (~138 GB weights at 4-bit → tight against 242 GB combined but feasible depending on KV/overhead)?
  (2) Does it hit the same `tml_fa4` paged-KV cliff as the 975B, or does its smaller KV make the
  per-step regather tolerable? (3) Does `LAMPORT_RS_SCONV=0` suffice on 2× RoCE, or does the
  2-node case differ? (4) Does MTP get past k=1 on the smaller model? Promote findings to the
  Inkling model page. Single source (announcement) → [conjecture].
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

## Forum-sourced open problems (2026-07-20 ingest)

- **[conjecture]** **Does the b12x backend enable arbitrary (non-power-of-2) TP on GB10?**
  (S-forum-6x-cluster, mclenithan): a 6× GB10 cluster reportedly runs TP=6 on "most models"
  via the b12x backend (lukealonso/b12x) — vLLM's stock distributed executor assumes
  powers-of-2 (2/4/8) for tensor parallel; 3-node previously required virtual-head padding
  (S-forum-3node-nccl, S-forum-mimo-3x). If b12x genuinely relaxes this constraint, it
  changes cluster sizing economics (6× CRS812 vs 8× needing CRS804). Hardware agent with
  ≥3 non-power-of-2 nodes should: (1) confirm TP=3 and TP=6 work on b12x without virtual-head
  padding, (2) measure whether non-power-of-2 TP incurs a collectives overhead vs the
  nearest power-of-2 (TP=4 vs TP=3, TP=8 vs TP=6), (3) verify the ~30 tok/s GLM-5.2 claim
  and whether all nodes actively compute or some only hold weights (unanswered in thread).
  No YAML/docker config was shared — the claim is unverifiable from the post alone.
  `[[wiki/multinode-tp-and-networking.md]]`

## Forum-sourced open problems (2026-07-20 ingest, Batch 25)

- **[conjecture]** **Paged-KV support for the `tml_fa4` Sm120/Sm121 cute FlashAttention path —
  the load-bearing blocker for rel-bias / FA4-arch models on GB10** (S-forum-inkling-nvfp4,
  greg190): the Inkling 8× Spark bring-up hit a steep long-context decode cliff (25 → 13.5 tok/s
  at c1 going from ~100 → 2048 ctx) because the `tml_fa4` Sm120 cute kernel has **no paged-KV
  support** — vLLM's paged cache can't feed it, so the workaround re-gathers the whole KV history
  into contiguous buffers every decode step (O(ctx)/token). This caps aggregate decode at ~24 tok/s
  at real context regardless of concurrency. Hardware agent / kernel dev should: (1) implement
  paged-KV reads natively in the Sm120/Sm121 cute FA4 kernel (the score-mod
  `vllm_flash_attn/cute` path is the intended sm12x route per the OP); (2) measure whether native
  paged-KV removes the cliff and recovers linear decode scaling with context; (3) verify the
  rel-bias q-row index clamp fix (vllm#49049) lands upstream. This would unblock not just Inkling
  but any rel-position-bias / FA4-arch model on sm_121a. See `[[wiki/attention-and-kv-cache.md]]`
  and `[[wiki/models/inkling.md]]`.
- **[conjecture]** **Intern-S2-Preview-397B on 4× DGX Spark — no quantization small enough for
  2× yet** (S-forum-intern-s2, chrm): `internlm/Intern-S2-Preview-397B` is a 397B preview model
  (Claude Opus-4.8 / GPT-5.5 class benchmarks claimed). **No quantization small enough for a 2×
  Spark cluster (242 GB) exists yet** — community requests a 4× Spark / AutoRound recipe. Open
  questions for a hardware agent: (1) At 4-bit, 397B ≈ ~200 GB weights → fits 4× Spark (484 GB)
  but not 2×; does an AutoRound NVFP4 quant fit 2× with tight KV? (2) Is the arch supported by
  vLLM on sm_121a, or does it need custom kernels? (3) What attention mechanism — does it hit the
  `tml_fa4` paged-KV cliff or the MoE cudagraph wall? No GB10-specific config, flags, or benchmarks
  reported yet — announcement only. Promote to a model page once a real bring-up with flags + tok/s
  lands. Single source → [conjecture].
- **[conjecture]** **MST sub-port splitting for switch-less 5-node GB10 mesh — verify on real
  hardware** (S-forum-kimi-k3-ceiling, mashie): the claimed technique to build a 5-node switch-less
  full mesh by splitting each QSFP port's 4×50G into 2×50G sub-ports via MST, yielding 6 RoCE
  interfaces per node. Hardware agent with 5+ nodes should: (1) confirm MST sub-port splitting
  works on the CX-7 and produces functional RoCE interfaces; (2) measure whether the half-bandwidth
  per sub-port (50G vs 100G) impacts TP=5 decode/prefill; (3) compare latency vs a MikroTik CRS812
  switch path. The OP is "currently working on" it — not yet verified. See
  `[[wiki/multinode-tp-and-networking.md]]`.

## Forum-sourced open problems (2026-07-21 ingest, Batch 26)

- **[conjecture]** **FlashInfer `sparse_mla_sm120` mbarrier livelock — needs upstream fix or
  independent reproduction** (S-forum-flashinfer-livelock, msunner): the FlashInfer
  `sparse_mla_sm120` kernels (prefill + decode) hard-wedge one rank GPU under cold-prefill load
  on GB10/sm_121. Root cause: mbarrier TRYWAIT phase check spin-loop (TMA/cp.async.bulk
  expect-tx accounting race). The Triton workaround (`FLASHMLA_SPARSE` + sm12x Triton patch) is
  validated (560+ clean sessions, no throughput penalty). Hardware agent / kernel dev should:
  (1) independently reproduce the livelock on a 4× Spark cluster with GLM-5.2 sparse MLA (the OP's
  config) — or a different sparse-MLA model — to confirm this is not a single-cluster hardware
  issue; (2) review the `sparse_mla_sm120` mbarrier expect-tx logic in
  `include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh`; (3) verify the Triton
  workaround has no quality regression vs FlashInfer on real sparse-MLA workloads; (4) file
  upstream FlashInfer issue with the cuda-gdb evidence. This is a load-bearing bug for any
  sparse-MLA model on GB10 (GLM-5.2, DeepSeek-V4-class, future MLA architectures). See
  `[[wiki/attention-and-kv-cache.md]]` → FlashInfer sparse-MLA livelock.
- **[conjecture]** **3-node PP vs TP=2 on 2 nodes — measure the overhead gap** (S-forum-3node-mesh,
  eugr, chunkai721): eugr claims 3-node PP is "roughly equivalent to a single Spark" (i.e. ~50%
  slower than 2-node TP=2). chunkai721's Qwen3.5-397B-A17B-int4-AutoRound benchmarks (12–14.4 tok/s
  decode) are consistent with single-node speed. Hardware agent should: (1) run the same model on
  1 node (if it fits) vs 2-node TP=2 vs 3-node PP and compare decode/prefill; (2) test LMCache on
  the 3rd node as dedicated KV cache — does it improve long-context throughput? (3) measure
  whether the NCCL mesh topology adds latency vs a switched topology for PP. Relevant for anyone
  with 3 Sparks deciding whether to use all 3 in PP or run 2+1 (TP=2 + sidecar). See
  `[[wiki/multinode-tp-and-networking.md]]` → Batch 26.

## Forum-sourced open problems (2026-07-22 ingest, Batch 29)

- **[conjecture]** **`nvidia-peermem` refuses to load on GB10 — is GPUDirect RDMA permanently
  unavailable, and is DOCA GPUNetIO/GDAKI the intended path?** (S-forum-6x-ring-rdma,
  alpaslan.erdag): `modprobe nvidia-peermem` fails with "Invalid argument" on kernel
  6.17.0-1021-nvidia with zero dmesg output; the .ko matches vermagic exactly. NCCL logs
  "GPU Direct RDMA Disabled" for all HCAs. NCCL loads `GIN_IB_GDAKI type 3` — suggesting
  DOCA GPUNetIO/GDAKI (GPU-initiated async via NVLink-C2C) may be the intended Grace-Blackwell
  GPU-NIC path, not classical PCIe P2P peermem. Hardware agent should: (1) confirm
  `nvidia-peermem` modprobe failure on their Spark (and check if a different kernel version
  or driver version changes it); (2) investigate whether DOCA GPUNetIO/GDAKI is available and
  functional on GB10 — is there a `gdaki` plugin or DOCA package to install? (3) if GDAKI works,
  measure whether it enables true GPU-NIC zero-copy and whether it changes the RDMA-vs-TCP
  gap from ~7% to something larger. This is the root question behind the proven "no GPUDirect"
  finding — *why* is it unavailable, and is there an alternative path? See
  `[[wiki/multinode-tp-and-networking.md]]` → Batch 29, `[[wiki/platform-gb10.md]]` → No GPUDirect RDMA.
- **[conjecture]** **NCCL_IB_SUBNET_AWARE_ROUTING availability — is it in NCCL main or only
  a fork?** (S-forum-6x-ring-rdma, alpaslan.erdag): the 6-node ring RDMA fix requires
  `NCCL_IB_SUBNET_AWARE_ROUTING=1`, which was used with a patched NCCL (not stock 2.28.9).
  Hardware agent should: (1) check if this flag exists in NCCL main / a recent release (≥2.30.4?);
  (2) if not, identify the fork/patch source and whether it's on track for upstream merge;
  (3) test on a 3+ node switchless topology to confirm it solves the channel→HCA round-robin
  topology-unawareness problem. This is the key blocker for >3-node switchless GB10 deployments
  using all 4 RoCE ports. See `[[wiki/multinode-tp-and-networking.md]]` → Batch 29.

## Forum-sourced open problems (2026-07-26 ingest, Batch 36)

- **[conjecture]** **Adaptive MTP (dynamic draft depth) — does the feedback loop overhead
  negate the benefit on bandwidth-bound Spark decode?** (S-forum-glm52-vision, CosmicRaisins):
  the adaptive MTP modification dynamically switches between 2 and 5 drafted tokens based on
  the acceptance rate of positions p2–p4 — theoretically matching draft depth to per-step
  acceptance (more tokens in code, fewer in prose). All existing MTP recipes on Spark use a
  fixed `num_speculative_tokens`. Hardware agent should: (1) benchmark adaptive-MTP vs
  fixed-MTP=2 and fixed-MTP=5 on GLM-5.2 (or any MTP-capable model) on a 4× Spark cluster;
  (2) measure whether the per-step acceptance-feedback overhead costs decode tok/s on the
  bandwidth-bound GB10 path; (3) verify the quality-vs-speed tradeoff claim (30+ tok/s in
  code without prose regression). See `[[wiki/engines.md]]` → GLM-5.2-Vision + adaptive MTP.

## Forum-sourced open problems (2026-08-01 ingest, Batch 46)

- **[conjecture]** **Inkling-Small FP8 KV cache — needs FlashAttention kernel modification,
  not a config toggle** (S-forum-inkling-small-2x, PILCOTHINK citing vLLM blog): Inkling uses
  BF16 for global attention; enabling FP8 KV requires modifying the Flash-attention kernel
  specifically used by Inkling. This caps 2× Spark context at ~300K (BF16 KV only), vs the
  model's 1M native window. Without FP8 KV, Inkling-Small is at a severe disadvantage vs
  DSV4-Flash (which supports NVFP4 KV) on the same hardware. Hardware agent / kernel dev
  should: (1) identify which FlashAttention kernel path Inkling routes through on sm_121a;
  (2) assess feasibility of adding FP8 KV support to that path; (3) benchmark context
  capacity gain (300K → ?) if FP8 KV is enabled. See
  `[[wiki/models/inkling.md]]` → Inkling-Small on 2× Spark.

## Forum-sourced open problems (2026-07-30 ingest, Batch 43)

- **[conjecture]** **CuTE DSL FP4 restriction to sm_100a — blocks Python-DSL NVFP4 kernels
  on GB10** (S-forum-sm121-support, baristankut/johnny_nv): CUTLASS C++ API works on sm_121,
  but the Python DSL (CuTE DSL) still restricts FP4 operations to sm_100a only (CUTLASS
  Issue #2800 open). This means any vLLM path using `CUTE_DSL_ARCH=sm_121a` for FP4 GEMM
  via the Python DSL hits a dispatch failure. The vLLM PR #29711 (device guard + runtime SM
  dispatch for `cutlass_scaled_fp4_mm`) is a workaround path. Hardware agent / kernel dev
  should: (1) verify whether CUTLASS v4.4.x resolves the DSL restriction; (2) test the
  PR #29711 device-guard path on a real NVFP4 model; (3) file or track Issue #2800 for
  sm_121a support. See `[[wiki/quantization-on-gb10.md]]`.
- **[conjecture]** **vLLM 0.14.0 — does it eliminate the --enforce-eager 20-30% perf loss on
  GB10?** (S-forum-sm121-support, johnny_nv): NVIDIA states vLLM 0.14.0 (expected shortly)
  "improves Blackwell compatibility and reduces reliance on eager execution." Hardware agent
  should: (1) benchmark a MoE model (e.g. MiMo-V2.5 or MiniMax-M3) on vLLM 0.14.0 vs 0.25.1
  with and without `--enforce-eager`; (2) measure whether cudagraph capture works for MoE
  on sm_121 in 0.14.0; (3) quantify the performance recovery. This would directly address
  the proven MoE cudagraph wall. See `[[wiki/cudagraphs-and-compile.md]]`.

## Forum-sourced open problems (2026-07-25 ingest, Batch 35)

- **[conjecture]** **PrismaQuant GridBook codebook quant — verify the native-dequant performance
  and quality claims on real GB10** (S-forum-gridbook, tenari/RobTand): the GridBook vLLM plugin
  claims codebook dequant constrained to the FP8/NVFP4 grid runs at full tensor-core speed with
  only ~10% decode / 30% prefill overhead, and that Qwen3.6-27B 5.5-bit achieves KL 0.0049 (77%
  lower than PrismaAura at the same rate). Hardware agent should: (1) load
  `rdtand/Qwen3.6-27B-prismaquant-codebook-5.5bit-vllm` on a Spark via the GridBook plugin and
  measure decode tok/s + prefill tok/s vs. the existing proven Qwen3.6-27B FP8 (~30 tok/s dense)
  and NVFP4 baselines; (2) verify the KL claim with a independent eval (ToolEvalBench or
  llama-benchy quality probes); (3) test the Hy3-295B-A21B 2.9-bit checkpoint — does it fit on a
  single 121 GB Spark and serve? At what tok/s? (4) test whether the MTP-head quant optimizer
  preserves spec-decode acceptance. This is the most promising single-Spark 300B-class-MoE path
  seen to date; verifying or refuting it is high-value. See `[[wiki/quantization-on-gb10.md]]` →
  PrismaQuant GridBook section.
- **[conjecture]** **Ant Ling-3.0-Flash 124B-A5B — re-ingest when weights drop, benchmark NVFP4
  and AutoRound INT4 on a single Spark** (S-forum-ling3-flash, entrpi): weights expected "after
  Aug 3." With 124B total at 4.5-bit NVFP4 ≈ 70 GB, or AutoRound INT4 ≈ 62 GB, this should fit
  comfortably on a single 121 GB Spark with room for a large KV pool. The 1/64 expert activation
  (5B active) suggests decode could be very fast — potentially exceeding Qwen3.5-122B-A10B
  (current single-Spark GOAT). The hybrid-linear KDA:MLA 5:1 attention is a new arch on GB10;
  verify whether vLLM/Atlas support it or if new attention kernels are needed. Hardware agent
  should: (1) re-ingest the forum thread when weights release; (2) quant to NVFP4 + AutoRound
  INT4; (3) benchmark decode/prefill/quality vs. Qwen3.5-122B-A10B and DSV4-Flash; (4) probe
  KDA attention kernel support. See `[[wiki/benchmarks.md]]` → Announced / upcoming models.

## Forum-sourced open problems (2026-08-03 ingest, Batch 50)

- **[conjecture]** **DSV4-Flash-0731 prefix cache unreliability on 2× Spark — isolate the
  cause** (S-forum-dsv4-0731-caching, Sa0lence): prefix cache on DSV4-Flash-0731 on 2-node Spark
  is non-deterministic — sometimes 1-2s prefill (cache hit), sometimes minutes to tens of
  minutes (cache miss), with no identifiable trigger. May relate to the known MTP+prefix-cache
  interaction bugs (S-forum-mtp-lossless) or to multi-node KV cache eviction under UMA memory
  pressure (S-forum-uvm-livelock). Hardware agent should: (1) run DSV4-Flash-0731 on 2× Spark
  with prefix cache ON, same prompt repeated 10×, log prefill time + cache hit/miss for each;
  (2) try with `MTP_NUM_TOKENS=0` (MTP off) to isolate MTP+cache interaction; (3) monitor
  `vllm` logs for cache eviction events; (4) try with `--gpu-memory-utilization 0.80` (more
  headroom) vs `0.90`. See `[[wiki/engines.md]]` → Batch 50 section.
- **[conjecture]** **Thermal sensor zone2/zone4 value swap — is it unit-specific or systemic?**
  (S-forum-powerstress, digiegg): the zone2/zone4 sensor swap anomaly (two ACPI thermal zones
  exchanging values over ~3s while zone0/zone5 sit at 97.6°C) persisted across EC + SoC firmware
  updates on one unit, suggesting a sensor mapping/calibration problem rather than a control-loop
  bug. RMA was approved for this unit. Hardware agent should: (1) run a 1Hz `acpitz` thermal
  sampler during sustained GPU load on a healthy Spark and check whether zone2/zone4 ever
  exchange values; (2) if yes, this is a systemic sensor mapping issue; if no, it was a
  unit-specific hardware fault. See `[[wiki/platform-gb10.md]]` → Batch 50 section.

## Forum-sourced open problems (2026-08-04 ingest, Batch 51)

- **[conjecture]** **FlashInfer sparse-MLA dispatch: does padding to 80 heads (16/rank, fast
  kernel) beat 65 heads (13/rank, generic kernel) at TP=5 on GB10?** (S-forum-glm52-3x-aqlm,
  karol.spark): the FlashInfer `_DECODE_DSV3_2_DISPATCH` table only instantiates
  `{8, 16, 32, 64, 128}` head counts. At TP=5, 13 heads/rank falls through to the generic
  paged-attention (1 tile, cheapest attention) but 9.4% MoE padding waste. Padding to 80 (16/rank)
  hits the fast specialized kernel but adds 25% ghost heads. The tradeoff (smaller GEMMs + generic
  kernel vs larger GEMMs + fast kernel) is unmeasured. Hardware agent with ≥5 nodes should:
  (1) run GLM-5.2 at TP=5 with 65-head and 80-head padding; (2) measure decode tok/s, GEMM time,
  attention time separately; (3) determine which wins. This generalizes to any non-power-of-2 TP
  on sparse-MLA models. See `[[wiki/attention-and-kv-cache.md]]` → FlashInfer dispatch table,
  `[[wiki/models/glm-5.2.md]]` → NVFP4+AQLM 3× section.
- **[conjecture]** **AQLM L1/L2 streaming kernel optimizations — isolate individual
  contributions** (S-forum-glm52-3x-aqlm, karol.spark): three env-gated kernel changes
  (`GLM_MOE_AQLM_CB=l1` + `GLM_MOE_AQLM_STREAM=1`, `GLM_NVFP4_STREAM=1`, draft cudagraph capture
  fix) landed together and produced +6.2% normalized decode. Each was microbenched individually
  (w13 +2.7%, w2 +22.5% for the AQLM changes; w13 +7% for NVFP4 stream) but not isolated in
  end-to-end decode. Hardware agent should: (1) run GLM-5.2 3× with each env var toggled
  independently; (2) measure end-to-end decode delta per change; (3) confirm whether the AQLM
  codebook L1 routing benefit generalizes to other AQLM-quantized models on GB10 (the L2 cache
  size vs codebook working set interaction is sm_121-specific). See
  `[[wiki/models/glm-5.2.md]]` → v3 kernel L1/L2 stream optimizations.

## Forum-sourced open problems (2026-08-14 ingest, Batch 69)

- **[conjecture]** **Silent idle hard lockup — LPI-3 deep-idle wake failure on GB10**
  (S-forum-idle-lockup, luis.poveda9321): a DGX Spark (ASUS GX10) reproducibly hard-locks
  at idle (~97% memory free, zero GPU workload) — the SoC descends into LPI-3 deepest
  idle state and never wakes. Zero forensic trace (no panic/OOM/Xid/hung_task). 7+
  occurrences; only happens at idle, never under load. Fourth distinct GB10 freeze
  mechanism. `hung_task_panic`/`softlockup_panic`/`panic_on_rcu_stall` armed + kdump
  active for next occurrence. Hardware agent should: (1) leave a Spark idle for extended
  periods with an out-of-band vitals logger (fsync'd every 3s, outside journald)
  capturing CPU idle-state trajectory (LPI levels) + PCIe ASPM state; (2) check if
  disabling deep idle states (`cpuidle.off=1` kernel param or `processor.max_cstate=1`)
  prevents the lockup; (3) try `pcie_aspm=off` to disable PCIe ASPM; (4) if kdump
  captures a vmcore, analyze the CPU/PCIe power-state transition path. This is the
  highest-priority platform stability issue for always-on deployments — a Spark that
  dies when idle between inference bursts is operationally worse than one that dies
  under load. See `[[wiki/platform-gb10.md]]` → idle LPI-3 lockup finding.


## First-party open problems (2026-08-24, S-dsv4-vision)

- **[proven]** **NVFP4 MoE has no serving path on this image.** Neither native selection nor the
  `VLLM_TEST_FORCE_FP8_MARLIN` fallback can serve an NVFP4-MoE checkpoint on sm_121 — the fallback
  dies in Marlin's FP4 **repack** with a PTX-toolchain rejection, before any inference. Worth
  retesting on each new image: it currently gates every NVFP4-MoE checkpoint, and it blocked the
  deciding experiment for grafted vision adapters. (`[[wiki/quantization-on-gb10.md]]`)
- **[conjecture]** **Does a grafted vision adapter work with the text half it was trained on?**
  webbrain-one's MoonViT adapter is blind on dealignai's MXFP4 CRACK weights, and its projector
  emits embeddings 26x the norm of even its *own* text half's embeddings. Upstream reports a
  passing image smoke test with an earlier text package, so the pairing is the leading suspect —
  untestable here until NVFP4 MoE serves. Weights staged. (`[[wiki/vision-adapters.md]]`)
- **[proven]** **Cudagraph replay IMA behind a multimodal wrapper class.** Capture succeeds, the
  first request faults in an unrelated text attention GEMM, and `--enforce-eager` is the only known
  workaround. Unexplained. (`[[wiki/cudagraphs-and-compile.md]]`)
