# sparkbase index

The map of the wiki. Every page, grouped, one line each. Start here. (Contract: [`SCHEMA.md`](SCHEMA.md).)

Every claim on these pages carries an **evidence tag** — `[conjecture]` `[reported]` `[reproduced]`
`[proven]` `[superseded]`. Build on `[proven]`; treat `[conjecture]` as "try it and tell us."

## Foundations
- **[Hardware-parity tenet](wiki/platform-gb10.md#foundational-tenet-hardware-parity-read-before-replicating-any-community-finding)** — DGX Spark is standardized; non-reproduction of a community finding is a software delta on your side, never an immutable hardware difference. Read before replicating any forum result.
- [platform-gb10](wiki/platform-gb10.md) — the hardware: sm_121/12.1a, 121 GB unified, ~270 GB/s (decode is bandwidth-bound), no native FP4/FP8-blockscale, no GPUDirect, OOM=reboot. **Read first.**
- [quantization-on-gb10](wiki/quantization-on-gb10.md) — what runs native (online-dynamic FP8) vs Marlin (FP4, block-scale FP8); ModelOpt-NVFP4; MXFP8/AWQ/AutoRound/GGUF; loader bugs.
- [cudagraphs-and-compile](wiki/cudagraphs-and-compile.md) — the two cudagraph walls (MoE on sm_121, cross-node host-staged NCCL / vllm#46253) and the "20 tok/s" math.
- [multinode-tp-and-networking](wiki/multinode-tp-and-networking.md) — CX7 twins + NCCL_IB_HCA + MTU 9000; no-ray TP; `--disable-custom-all-reduce`; mDNS/sshd ops; why cross-node is slow.
- [attention-and-kv-cache](wiki/attention-and-kv-cache.md) — TRITON_ATTN / DIFFKV / FLASHINFER selection; block-size 128 for MSA; fp8 KV; ViT JIT; gemma-norm ICE.

## Engines & tooling
- [engines](wiki/engines.md) — vLLM vs Atlas vs llama.cpp; Atlas internals (AOT kernels, KV sizing, MTP, loader bugs); durable serving pattern.
- [containers-and-tooling](wiki/containers-and-tooling.md) — known images & what they load; probing tricks; std env; Xet/permission/io_uring gotchas; ComfyUI flags.
- [llama-cpp-rpc](wiki/llama-cpp-rpc.md) — GGUF + 2-node pipeline RPC; `--no-mmap` mandatory; sm_121a build; tensor-split.

## Models
- [mimo-v2.5](wiki/models/mimo-v2.5.md) — 310B Omni MoE NVFP4+MXFP8 DiffKV; mods chain; abliteration-is-damaged diagnostic.
- [minimax](wiki/models/minimax.md) — M2.7 AWQ daily-driver (~24 tok/s); M3 428B MSA+vision cross-node (~5 tok/s, walled).
- [holo-3.1](wiki/models/holo-3.1.md) — computer-use VLM (Qwen3.5 VL MoE); NVFP4 wins; thinking-OFF = 4.2×.
- [gemma-4](wiki/models/gemma-4.md) — 12B unified arch (image support = serveability); FP8 online-dynamic 2× fast path.
- [diffusiongemma](wiki/models/diffusiongemma.md) — 26B-A4B block-diffusion LLM; native in vllm-node; NVFP4 MoE needs VLLM_TEST_FORCE_FP8_MARLIN; bf16 deployed, NVFP4 retired.
- [qwen](wiki/models/qwen.md) — 3.5/3.6/Coder-Next; MoE-A3B NVFP4+MTP ~142 tok/s vs dense ~30; Atlas loader landmines.
- [nemotron-3](wiki/models/nemotron-3.md) — hybrid Mamba-2 MoE; 120B Q8 via llama.cpp RPC; Nano-Omni vision/omni single-node.
- [step-3.7](wiki/models/step-3.7.md) — retired; kept for the MTP-needs-cudagraphs finding.

## Reference
- [benchmarks](wiki/benchmarks.md) — collated decode tok/s + concurrency table; append rows.
- [roadmap](wiki/roadmap.md) — open problems & areas of further development.
- [sources](sources/README.md) — where findings came from (`S-` ids, source-typed).
- [log](log.md) — append-only ingest/change log.

## Forum ingest 2026-07-08
- 184 total NVIDIA DGX Spark forum threads processed (20 high-priority + 164 remaining from main + projects forums).
- See `sources/README.md` → Forum sources (Batch 1 + Batch 2) for all registered sources.
- Pages touched: platform-gb10 (power wedge, TMA, thermal, GSP timeout, drivers, display/USB/WiFi issues),
  quantization-on-gb10 (AWQ vs NVFP4, MXFP4, CUTLASS failure, distributed quant),
  multinode-tp-and-networking (CX-7 PCIe, MikroTik switches, 2D parallelism, DDP, SKU mixing),
  engines (Atlas, ds4/DwarfStar, DFlash block-spec), containers-and-tooling (community tools),
  models/mimo-v2.5 (TP=3 virtual-head padding, community recipes), models/minimax (4× recipes,
  llama.cpp RPC, MSA, AWQ/MXFP4/NVFP4-KV variants), benchmarks (21 forum-reported rows),
  llama-cpp-rpc (M3 RPC), roadmap (4 new open problems).

## Forum ingest 2026-07-09
- 160 new NVIDIA DGX Spark forum threads processed (~48 new sources, Batch 3).
- See `sources/README.md` → Batch 3 forum sources for all registered sources.
- Pages touched: platform-gb10 (CX-7 bricking, SDPA corruption, SageAttention, vLLM 26.06 broken,
  OOM hang fix, fwupd mismatch, UMA bandwidth, torchaudio), multinode-tp-and-networking (NCCL 2.30.4,
  SGLang RDMA passthrough, SGLang traps, CUTLASS MoE OOM, 4-node mesh, MTP NEXTN),
  quantization-on-gb10 (KVarN, Spark Auto Round, KV benchmarks, TurboQuant, STREAM LOADING,
  ModelOpt CPU-bound, vLLM regression, MTP math, heterogeneous quant), engines (DDTree, STREAM LOADING,
  SM121 kernel guide, vLLM regression), containers-and-tooling (Nunchaku, ComfyUI, llama.cpp container,
  QAT models, Mistral OOM, torchaudio), models/mimo-v2.5 (DFlash 22→67, v0.24.0 DFlash+NVFP4 KV),
  models/minimax (W4A16-GPTQ corroborated, M2.5 4× SGLang), models/nemotron-3 (Super MTP, Ultra 550B,
  ABI fix), benchmarks (17 new forum-reported rows), roadmap (4 new open problems).

## Forum ingest 2026-07-10
- 10 new forum topics found (8 technically relevant, 2 skipped as social/buying advice).
- 7 new sources registered (Batch 4). 8 topic IDs added to processed_topics.txt (total now 344).
- Pages touched: quantization-on-gb10 (FLUX.2 NVFP4 W4A4 compute, weight-only vs activation-quantized),
  platform-gb10 (UMA mmap double-allocation OOM workaround, TCG OPAL/UEFI corruption, display controller
  pixel clock, ONNX GPU discovery), containers-and-tooling (llama-benchy, cluster dashboard, Sunshine
  RDP, FLUX.2 Images-API server), benchmarks (3 new forum-reported rows: MiniMax-M2.1, GLM-4.7-Flash,
  FLUX.2-dev image gen), models/mimo-v2.5 (topic 375923 already ingested in Batch 3).

## Forum ingest 2026-07-10 (Batch 5)
- 3 new forum topics found, all technically relevant.
- 3 new sources registered (Batch 5). 3 topic IDs added to processed_topics.txt (total now 349).
- Pages touched: models/mimo-v2.5 (SGLang 4× FP8 recipe, MTP OOM, NVFP4 MoE backend gap on SM121a,
  sampling params/Thought Loop mitigation), models/minimax (M2.7 NVFP4/AWQ/FP8 recipes on 2×/4× Spark,
  FlashInfer-CUTLASS beats CUTLASS, AWQ beats NVFP4 corroborated by 3 independent sources → [reported],
  Unsloth FP8 4× 36 tok/s), benchmarks (4 new LLM forum rows + 6 diffusion model image gen rows),
  quantization-on-gb10 (FlashInfer-CUTLASS stability, diffusion weight-only vs activation-quantized
  NVFP4 distinction), containers-and-tooling (DIFFUSERS_ATTN_BACKEND=_native_cudnn speedup,
  diffusion model benchmark table).

## Forum ingest 2026-07-11 (Batch 6)
- 3 new forum topics found, all technically relevant.
- 3 new sources registered (Batch 6). 3 topic IDs added to processed_topics.txt (total now 352).
- Pages touched: platform-gb10 (random shutdowns — thermal paste degradation, CPU hot-spot sensor
  blind spot, PDU fault variant, no WoL, Nsight Systems sudo requirement), multinode-tp-and-networking
  (3-node ring topology NCCL failure, cable mixing MTU mismatch, explicit SSH resolution for >2 nodes).

## Forum ingest 2026-07-12 (Batch 8)
- 1 new forum topic found, technically relevant.
- 1 new source registered (Batch 8). 1 topic ID added to processed_topics.txt (total now 355).
- Pages touched: platform-gb10 (GPU clock wedge follow-up — 5 min wait sufficient [reported],
  power-drain method [conjecture], PSU root-cause hypothesis [conjecture]).

## Forum ingest 2026-07-12 (Batch 9)
- 3 new forum topics found (2 technically relevant, 1 skipped as non-GB10-specific).
- 2 new sources registered (Batch 9). 3 topic IDs added to processed_topics.txt (total now 358).
- Pages touched: platform-gb10 (reboot power-cycle completion failure [conjecture]),
  multinode-tp-and-networking (CX-7 dual setup field report — third-party DAC, TCP ceiling ~16 Gb/s,
  DCGM on GB10, PSI OOM alerting, NetworkManager config, cluster tax metric).

## Forum ingest 2026-07-13 (Batch 10)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 10). 2 topic IDs added to processed_topics.txt (total now 360).
- Pages touched: quantization-on-gb10 (NVIDIA refreshed NVFP4 recipe, Qwen3/Qwen3.6 27B
  TensorRT-LLM errors [conjecture]), containers-and-tooling (nvidia-vfx no aarch64 wheel,
  NVIDIA confirmed no plans, ComfyUI RTX nodes broken [reported]).

## Forum ingest 2026-07-14 (Batch 12)
- 4 new forum topics found (3 technically relevant, 1 skipped as buying advice).
- 3 new sources registered (Batch 12). 4 topic IDs added to processed_topics.txt (total now 368).
- Pages touched: engines (TokenSpeed SM12x-stable engine — prefill +10-14% vs vLLM, decode behind
  70-74%, KV +25%, NCCL 2.30.4 mandatory, build recipe), containers-and-tooling (Spark Studio
  dashboard — live UMA monitor, pre-launch memory guard, agent auto-fix), attention-and-kv-cache
  (DSV4-Flash KV cache ~15 GB/1M tokens/node, CUDA graph memory profiling overhead),
  benchmarks (2 new forum-reported rows: DSV4-Flash TokenSpeed vs vLLM fork).

## Forum ingest 2026-07-14 (Batch 13)
- 1 new forum topic found (technically marginal — non-LLM model recommendation).
- 1 new source registered (Batch 13). 1 topic ID added to processed_topics.txt (total now 369).
- Pages touched: sources/README, log, index. No wiki page edits.
- Finding: ACE-Step v1.5 XL (music generation) runs on single Spark, fits comfortably in VRAM
  with companion 5Hz-LM-4B lyrics model. 3 independent users confirm (danielgbates, joey28,
  aostang) → would be [reported] if it warranted a page. Not ingested to wiki because: outside
  core LLM-inference scope (not vLLM/llama.cpp/sglang), no GB10-specific flags/env vars/errors/
  tok-s numbers/quant formats, and "fits in 121 GB" is trivially true for a single-model workload
  with no quantization. Source registered for provenance; no wiki page created.

## Forum ingest 2026-07-15 (Batch 14)
- 6 new forum topics found (4 technically relevant, 2 skipped: robotics RT kernel, RMA complaint).
- 4 new sources registered (Batch 14). 6 topic IDs added to processed_topics.txt (total now 375).
- Pages touched: models/qwen (Unsloth NVFP4 ~15% slower than nvidia on GB10 [reported] via 3
  independent benchmarks; flashinfer_b12x unavailable on stock vLLM; working Marlin+MTP recipe;
  W4A16 bypass hypothesis; quality parity), quantization-on-gb10 (Unsloth NVFP4 slower on GB10,
  flashinfer_b12x gap, W4A16 bypass), benchmarks (5 new forum-reported rows: nvidia vs Unsloth
  Qwen3.6-35B-A3B NVFP4 comparison), engines (multi-model co-hosting — vision+LLM on 2× Spark
  memory-starved, offload vision to separate machine [reported], multimodal front-end pipeline),
  platform-gb10 (HPC/slurm CPU P/E core topology, CX-7 switch config, Llama 3.2 3B finetune
  8× slower than benchmark — known FAQ).

## Forum ingest 2026-07-15 (Batch 15)
- 4 new forum topics found (3 technically relevant, 1 skipped: buying advice).
- 3 new sources registered (Batch 15). 4 topic IDs added to processed_topics.txt (total now 379).
- Pages touched: platform-gb10 (CX-7 hot-pluggable ports — not visible until cable connected,
  /etc/nvidia/cx7-hotplug-enabled, idle power doubles when active), engines (LLM + ComfyUI
  co-hosting — vLLM KV cache starves UMA, --gpu-memory-utilization 0.7-0.8, llama.cpp better
  for co-hosting, swapoff -a), multinode-tp-and-networking (~23 GB/s cross-node vs ~600 GB/s
  in-box bottleneck, MoE gains flatten past TP=4, FP8 training impossible on sm_121, Megatron
  caveats), models/qwen (Qwen3.5-397B on 8× GB10 31-35 tok/s, architecture comparison),
  benchmarks (1 new row: Qwen3.5-397B FP8 8× GB10).

## Forum ingest 2026-07-13 (Batch 11)
- 4 new forum topics found (2 technically relevant, 2 skipped as non-technical).
- 2 new sources registered (Batch 11). 4 topic IDs added to processed_topics.txt (total now 364).
- Pages touched: engines (easy-vllm code-agent harness, DSV4-Flash GB10 via jasl/vllm SM12x fork,
  torch 2.11+ ABI wall, MXFP4 MoE→MARLIN→UMA OOM, mem_watchdog+earlyoom),
  multinode-tp-and-networking (4-node CRS504 100G switch — 5-10% PP loss, zero decode loss,
  ~13 Gb/s measured traffic, $25 100G cable works), benchmarks (3 new forum rows: DSV4-Flash
  TP=4/TP=2, M3-AWQ+EAGLE TP=4), containers-and-tooling (easy-vllm tool),
  models/minimax (M3-AWQ+EAGLE on CRS504 corroborates existing benchmarks).

## Forum ingest 2026-07-16 (Batch 16)
- 3 new forum topics found (2 technically relevant, 1 skipped as buying advice/warranty/social).
- 2 new sources registered (Batch 16). 3 topic IDs added to processed_topics.txt (total now 382).
- Pages touched: engines (Colibri — pure-C expert-streaming engine for GLM-5.2 744B MoE on
  single Spark, 2.4-3.3 tok/s, O_DIRECT 9.69 GB/s, CACHE_ROUTE experimental routing),
  containers-and-tooling (ComfyUI Docker optimized for DGX Spark — CUDA 13.1 sm_121, SageAttention 2,
  double-VRAM bug fix copy=False + --disable-mmap, --disable-dynamic-vram, cudaMemGetInfo
  under-reports free UMA when co-resident CUDA process — psutil fix),
  platform-gb10 (cudaMemGetInfo under-reports free memory on UMA with co-resident process
  [conjecture]), benchmarks (1 new forum-reported row: GLM-5.2 744B Colibri expert streaming),
  roadmap (Colibri demonstrates expert-streaming approach in practice — bottleneck is attention
  not disk I/O).

## Forum ingest 2026-07-16 (Batch 17)
- 5 new forum topics found (4 technically relevant, 1 skipped: 376589 = buying advice "triple
  stack").
- 4 new sources registered (Batch 17). 5 topic IDs added to processed_topics.txt (total now 387).
- Pages touched: quantization-on-gb10 (NVFP4 meta-analysis — NVFP4 leaves ~half layers bf16 vs
  Int4 all-layers; TRT-LLM NVFP4 slower than GGUF Q4_K_M; bandwidth efficiency 42-48%; NVFP4 now
  operational via community Docker; FlashInfer 0.6.8.1 improvements [reported]),
  models/nemotron-3 (Ollama v0.30.x-v0.31.2 parser regression breaks Nemotron-3-Super on GB10,
  fix: downgrade to 0.24.0 [conjecture]; NVFP4 bandwidth efficiency 42-48% [reported]),
  multinode-tp-and-networking (ib_write_bw falsely reports >64 KiB RDMA WRITE failure on GB10 —
  fabric is fine; NCCL_NET_PLUGIN=none, NCCL_TOPO_FILE correction, RoCE NIC-offloaded counters,
  arp_ignore=1/arp_announce=2 [conjecture]),
  platform-gb10 (Ollama parser regression [conjecture], NVFP4 bandwidth efficiency 42-48%
  [reported]), engines (DSV4-Flash-DSpark-Abliterated source added),
  benchmarks (5 new forum-reported rows: Llama-3.3-70B NVFP4 vs GGUF Q4_K_M, Nemotron-3-Super
  NVFP4 1×/2×, DSV4-Flash-DSpark-Abliterated 50-60 tok/s).

## Forum ingest 2026-07-17 (Batch 18)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 18). 2 topic IDs added to processed_topics.txt (total now 389).
- Pages touched: benchmarks (3 new [conjecture] rows — GLM-5.2-Int4-Int8Mix on 8× GB10 TP8 DCP=1
  ~1,200 t/s prefill / 33–54 t/s decode; TP4+PP2 ~12 t/s MTP collapse; DCP4 decode-starvation
  scheduler), multinode-tp-and-networking (NCCL_BUFFSIZE 16 MB at TP8 [conjecture], TP4+PP2 wrecks
  MTP acceptance → 8% [conjecture], DCP4 decode starvation + decode-aware prefill scheduler
  [conjecture], draft_tensor_parallel_size=1 [conjecture]), quantization-on-gb10 (b12x W4A8 MoE
  backend — INT4 weights + INT8 activations via native FP8 CUTLASS [conjecture]; stale
  topk_indices_buffer in flashinfer SM120 sparse MLA PR#46994 [conjecture]; quantized NextN draft
  token mapping [conjecture]; VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 [conjecture]; Int4-Int8 mix
  quant [conjecture]), models/qwen (Bonsai 27B binary/ternary Qwen3.6-27B — hypothesis: faster
  decode on bandwidth-bound Spark dense, no GB10 benchmarks yet [conjecture]), roadmap (3 new
  open problems: v16+b12x W4A8 isolated contribution, Bonsai sm_121 kernel path, DCP4 scheduler
  on DCP1).

## Forum ingest 2026-07-17 (Batch 19)
- 5 new forum topics found, all technically relevant (USB2 fallback, firmware updates, OTA
  loop, ASUS GX10 firmware, host freeze during TP=2 prefill).
- 5 new sources registered (Batch 19). 5 topic IDs added to processed_topics.txt (total now 394).
- Pages touched: platform-gb10 (USB3 SuperSpeed PHY not registered → USB2 fallback [reported]
  via 7 independent users; MediaTek T-PHY no ACPI binding; new FE firmware EC/UEFI versions;
  DGX Dashboard OTA loop + nvidia-spark-ota-check diagnostic tool; ASUS GX10 v0103 PD firmware
  fixes thermals ~8-10 W lower [reported] + 4× link speed [conjecture]; total host freeze
  during heavy TP=2 prefill = thermal shutdown with zero forensic trace [conjecture]),
  multinode-tp-and-networking (ASUS PD firmware 4× link speed [conjecture]; host freeze
  during prefill = highest combined SoC power scenario [conjecture]),
  sources/README, index, log.

## Forum ingest 2026-07-11 (Batch 7)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 7). 2 topic IDs added to processed_topics.txt (total now 354).
- Pages touched: models/mimo-v2.5 (detailed 2-node renek recipe — driver KV pool diff, NCCL CGA buffer,
  MTP acceptance rates, enforce-eager at 160K, full env/config, 30-33 tok/s; tonyd615 repo 38 tok/s
  non-eager; synthetic vs real-world gap), platform-gb10 (first-boot WiFi SSID not broadcasting,
  QR→product page, monitor+keyboard workaround), attention-and-kv-cache (TRITON_ATTN_DIFFKV
  quantized KV guard), multinode-tp-and-networking (NCCL v2.30u1 CGA buffer), benchmarks (1 new
  forum-reported row: MiMo-V2.5-NVFP4 renek recipe 30-33 tok/s).
