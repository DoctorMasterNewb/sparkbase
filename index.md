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
- [mistral-small-4](wiki/models/mistral-small-4.md) — 119B NVFP4 MLA; TRITON_MLA resolves head_size=320 on SM121; ~28-33 tok/s [reported]; Eagle/MTP not working; --shm-size 16g kernel crash.
- [step-3.7](wiki/models/step-3.7.md) — retired; kept for the MTP-needs-cudagraphs finding.
- [laguna-s-2.1](wiki/models/laguna-s-2.1.md) — 117.6B MoE NVFP4 single-node; DFlash spec=7; 22.6 tok/s decode, flat across depths; **retired** — output quality below MiMo/DeepSeek, no speed advantage over Qwen3.6.
- [inkling](wiki/models/inkling.md) — Thinking Machines multimodal MoE (975B/41B-active); NVFP4 on 8× Spark, paged-KV cliff, Lamport-on-RoCE escape hatch, kernel bugs filed.
- [glm-5.2](wiki/models/glm-5.2.md) — Zhipu AI 744B/40B-active MoE (sparse-MLA); 4×–8× Spark recipes, hybrid FP8+NVFP4+MXFP4 quant, MTP quality, reasoning-parser bug, KV kernel constraints.

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

## Forum ingest 2026-07-18 (Batch 20)
- 1 new forum topic found, technically relevant (MTP lossless? — quality & prefix-cache bugs).
- 1 new source registered (Batch 20). 1 topic ID added to processed_topics.txt (total now 395).
- Pages touched: engines (MTP measurably affects output quality [conjecture] — tool-call bench ~5
  pts; ~40% speed vs ~2% quality hit on Qwen3.6-27B [conjecture]; vLLM+llama.cpp MTP+prefix-cache
  interaction bugs [conjecture]; DS4F prefix-batch 16384/MTP=4 → 70-75% acceptance [conjecture];
  "theory != deployment" practical-lossiness debate [conjecture]), roadmap (new open problem:
  measure MTP quality impact & prefix-cache interaction on real Spark).

## Forum ingest 2026-07-19 (Batch 22)
- 3 new forum topics found, all technically relevant (EC firmware fan-curve regression, Nemo-RT
  voice agent, LiteLLM multi-model orchestrator).
- 3 new sources registered (Batch 22). 3 topic IDs added to processed_topics.txt (total now 400).
- Pages touched: platform-gb10 (EC firmware 0x0300xxxx breaks fan curve → 96-97°C ACPI zones,
  inaudible fans; EC isolates fan control from OS — fancontrol/pwmconfig/nvidia-settings can't
  override; fwupdmgr downgrade to 0x02004e18 fix; idle 60→32°C, load 35-37°C, 0% throttling,
  120-125W/node @ 95% GPU util; first reported EC firmware *regression* on Spark; fan control is
  EC-isolated, not OS-overridable [conjecture]; relationship to 0x03000508 "improves EC" update
  unresolved), containers-and-tooling (harinezumigel-llm-stack LiteLLM+vLLM orchestrator for
  single-Spark multi-model lifecycle management [conjecture]; thread surfaces sparkstation
  (kshetrajna12/sparkstation); Nemo-RT Community voice agent — VAD+STT+LLM(Qwen3-8B-FP8 via
  vLLM)+TTS on one GPU, OpenAI Realtime API-compatible, ~20 concurrent calls on Spark, native
  FP8 + arm64 build [conjecture]).

## Forum ingest 2026-07-19 (Batch 23)
- 3 new forum topics found, all technically relevant (NVIDIA Sync locale bug, ASUS GX10 thermal
  throttling corroborating the EC fan-curve regression, Inkling 975B/276B MoE announcement).
- 3 new sources registered (Batch 23). 3 topic IDs added to processed_topics.txt (total now 403).
- **Evidence promotion:** the EC firmware fan-curve regression (S-forum-ec-fan-rollback,
  originally [conjecture] in Batch 22) is **promoted to [reported]** — independently corroborated
  on a 3rd OEM SKU (ASUS GX10, S-forum-ec-fan-asus) with the same symptom fingerprint (ACPI zones
  96.6°C, fans N/A, SW/HW thermal slowdown counters). Three OEM SKUs now agree: Gigabyte, MSI FE,
  ASUS GX10.
- Pages touched: platform-gb10 (new Batch 23 section — ASUS GX10 thermal throttling corroborates
  EC fan-curve regression → [reported]; root-cause narrows from EC table to SoC/UEFI interaction
  via byte-identical fan-curve comparison 48%@85°C/54%@93°C/68%@95°C/100%@97°C [conjecture];
  first published GB10 fan-curve bytes [conjecture]; fwupdmgr downgrade unavailable for ASUS GX10
  [conjecture]; dgx-spark-fieldiag 2.0.4-1 ofed-scripts packaging bug [conjecture]; existing
  Batch 22 entry promoted [conjecture]→[reported] with corroboration note),
  multinode-tp-and-networking (NVIDIA Sync / Cluster Assistant fails "Software version" check on
  non-English locale — apt-cache policy parser breaks on localized "Installiert:"/"Installé :"
  labels → false "System Software Update Required"; workaround sudo update-locale
  LC_MESSAGES=en_US.utf8; hotfix pending [conjecture]), roadmap (3 new open problems: Inkling
  975B/276B MoE bring-up not yet characterized, EC fan-curve root-cause isolation, fieldiag
  ofed-scripts dependency gap).

## Forum ingest 2026-07-20 (Batch 24)
- 1 new forum topic found, technically relevant (6× GB10 cluster via MikroTik CRS812, b12x TP=6).
- 1 new source registered (Batch 24). 1 topic ID added to processed_topics.txt (total now 404).
- Pages touched: multinode-tp-and-networking (6× GB10 cluster via CRS812 — b12x backend enables
  non-power-of-2 TP=6 on most models; GLM-5.2 ~30 tok/s single-stream; cluster 800-1180 W peak;
  consistent with sublinear scaling between TP=4 and TP=8 [conjecture]), benchmarks (1 new
  [conjecture] row: GLM-5.2 6× TP=6 ~30 tok/s), roadmap (new open problem: does b12x enable
  arbitrary non-power-of-2 TP on GB10 — virtual-head padding previously required for TP=3).

## Forum ingest 2026-07-21 (Batch 26)
- 4 new forum topics found, all technically relevant.
- 4 new sources registered (Batch 26). 4 topic IDs added to processed_topics.txt (total now 408).
- **Headline finding:** FlashInfer `sparse_mla_sm120` mbarrier livelock on GB10/sm_121 — root-caused
  via cuda-gdb (mbarrier TRYWAIT spin-loop under cold-prefill), validated Triton workaround
  (FLASHMLA_SPARSE + sm12x patch, 560+ clean sessions, no throughput penalty). Major kernel bug
  for any sparse-MLA model on GB10. Flagged for priority hardware verification in roadmap.
- Pages touched: attention-and-kv-cache (FlashInfer livelock + Triton workaround),
  multinode-tp-and-networking (3-node full-mesh guide — CX-7 triangle, cross-connect port0↔port1,
  TP requires power-of-2, 3-node PP ~single-node speed, PP+MTP not supported, LMCache for KV
  node, NCCL mesh merged to main, fastsafetensors freeze, gpu_memory_utilization 0.8 for PP),
  containers-and-tooling (community images lag upstream vLLM 0.25.1/NCCL 2.30.7),
  platform-gb10 (EC firmware 0x00000500→0x00000507 silent failure, fwupdmgr get-results diagnostic),
  benchmarks (2 new [conjecture] rows: Qwen3.5-397B-A17B 3-node PP decode 12–14.4 tok/s + prefill
  912–1242 tok/s), roadmap (2 new open problems: FlashInfer livelock reproduction/fix, 3-node PP
  vs TP=2 overhead measurement).

## Forum ingest 2026-07-21 (Batch 27)
- 2 new forum topics found (1 technically relevant, 1 skipped as non-technical).
- 1 new source registered (Batch 27). 2 topic IDs added to processed_topics.txt (total now 410).
- Pages touched: platform-gb10 (sysfs thermal zone layout under load — zones 0/5 hottest at
  94.6 °C, GPU ~10 °C cooler than CPU; `tegrastats` Jetson Orin Nano binary works on GB10;
  GPU clock capping as thermal mitigation per wildpines.ai blog). All [conjecture].
- Skipped: topic 377428 (AirLLM theoretical parameter-ceiling speculation — no runs, no
  measurements, replies all jokes; adds nothing beyond known 128 GB unified memory / 4 TB SSD
  facts already in platform-gb10 and the Colibri expert-streaming approach in engines).

## Forum ingest 2026-07-22 (Batch 28)
- 10 new forum topics found (4 technically relevant, 5 skipped as social/buying/speculation,
  1 already covered by existing source).
- 4 new sources registered (Batch 28). 10 topic IDs added to processed_topics.txt (total now 420).
- Pages touched: platform-gb10 (UVM page-migration livelock — hard shutdown under sustained load,
  --gpu-memory-utilization 0.85-0.92 fix, platform firmware update, power cap, clock lock;
  GB10B scanout carveout allocation failure in Sway at 6K resolution; RealSense D435 USB
  disconnect fixed by July firmware), containers-and-tooling (sparkDash by MiaAI-Lab — second
  independent multi-Spark monitoring dashboard). All [conjecture].
- Skipped: 377602 (Motif-3-Beta model announcement, no GB10 specifics), 377626 (best model for
  3-node, pointer to existing M3 TP=3 docs), 372722 (buy vs rent, buying advice), 364493
  (Windows 11 ARM installation, not LLM inference), 377396 (Qwen 3.8 launch speculation).
- 376643 (Sparkrun webui by brainchillz) already covered by existing S-forum-sparkdash — same
  repo (brainchillz/sparkdash), just a different forum post. Marked processed, no new source.

## Forum ingest 2026-07-23 (Batch 30)
- 3 new forum topics found, all technically relevant.
- 3 new sources registered (Batch 30). 3 topic IDs added to processed_topics.txt (total now 431).
- **Headline finding:** Mistral Small 4 119B NVFP4 on DGX Spark — 67-post thread with working recipe.
  MLA head_size=320 rejected by all stock backends on SM121; TRITON_MLA (via eugr's spark-vllm-docker)
  resolves it. ~28-33 tok/s decode corroborated by 5 independent forum users → [reported]. Known
  issues: reasoning_effort bug (vLLM 0.17.2rc1), tool-calling leaks (PR #39217), Eagle/MTP not
  working, --shm-size 16g causes kernel crash (must use 4g). vLLM 0.25.1 now publishes native arm64
  images — no custom build needed for base vLLM.
- Pages touched: models/mistral-small-4 (NEW — full model page), models/qwen (Qwen3.6-35B-A3B FP8
  2× recipe 75-80 tok/s [conjecture]), platform-gb10 (CX7 DAC thermal penalty 6°C even after
  software disable — only physical removal works; dgx-spark-mlnx-hotplug udev/ACPI mechanism
  [conjecture]), benchmarks (6 new forum-reported rows: 5 Mistral Small 4 + 1 Qwen3.6 FP8),
  sources/README, index, log.

## Forum ingest 2026-07-22 (Batch 29)
- 4 new forum topics found, all with at least marginal GB10 relevance.
- 4 new sources registered (Batch 29). 4 topic IDs added to processed_topics.txt (total now 428).
- **Headline finding:** 6-node DGX Spark ring topology (S-forum-6x-ring-rdma) — the most
  technically dense multinode thread in weeks. Three major findings: (1) RoCE RC QPs require
  L2 adjacency — routed L3 RDMA fails at the ibv_modify_qp verbs layer for non-adjacent ring
  pairs, explaining why official topologies stop at 3-node full-mesh; (2) NCCL_IB_MERGE_NICS=0
  + NCCL_IB_SUBNET_AWARE_ROUTING=1 (patched NCCL) together fix 6-node ring RDMA — stock NCCL's
  round-robin channel→HCA assignment is not topology-aware; (3) nvidia-peermem refuses to
  insert ("Invalid argument") on GB10 — GPUDirect RDMA unavailable, DOCA GPUNetIO/GDAKI may be
  the intended path. First quantified RDMA-vs-TCP comparison: only ~7% gain (both host-staged).
- Pages touched: multinode-tp-and-networking (7 new [conjecture] findings — RoCE L2-adjacency,
  MERGE_NICS=0+SUBNET_AWARE_ROUTING fix, NCCL channel topology-unawareness, GID asymmetry, TCP
  fallback workaround, nvidia-peermem modprobe failure + GDAKI hypothesis, Hunlx 3-node env
  recipe), platform-gb10 (3 new [conjecture] — UEFI firmware update stepping-stone requirement,
  serial console not supported, sleep/suspend disabled by default), benchmarks (2 new
  [conjecture] rows — Qwen3.6-35B-A3B NVFP4 6-node PP=6 TCP 326 / RDMA 349 tok/s aggregate).
- All [conjecture] — single forum source each. No new wiki pages created.

## Forum ingest 2026-07-23 (Batch 31)
- 2 new forum topics found (1 technically relevant, 1 skipped as model announcement).
- 1 new source registered (Batch 31). 2 topic IDs added to processed_topics.txt (total now 433).
- Pages touched: containers-and-tooling (spark-vllm-docker build flags — `--rebuild-vllm`
  forces local rebuild, `--use-wheels` uses prebuilt wheels, repo builds from `main` with no
  pinned vLLM version — all [conjecture]), models/mistral-small-4 (build flags detail added
  to existing spark-vllm-docker section [conjecture]), sources/README, index, log.
- Skipped: 377762 (Motif-3 Beta model announcement, no GB10 specifics — same model already
  skipped in Batch 28 as topic 377602).

## Forum ingest 2026-07-24 (Batch 32)
- 7 new forum topics found (4 technically relevant, 3 skipped: social, RMA, entitlement).
- 4 new sources registered (Batch 32). 7 topic IDs added to processed_topics.txt (total now 440).
- **Headline finding:** MiniMax-M3 NVFP4 TP=3 on 3× DGX Spark (S-forum-m3-tp3, tonyd615) —
  first working TP=3 recipe via Luke Alonso's chthonic vLLM+b12x virtual sharding commit.
  Three undocumented head-node OOM fixes (safetensors load format, Ray object-store cap,
  Ray memory monitor disable). NCCL LD_PRELOAD shim trap — baked container shim silently
  overrides user-installed NCCL 2.30u1. Cold power-drain fixes stuck ib_write_bw (12.8→111.85
  Gb/s). TP=3 bandwidth fix increases concurrency, not single-stream tok/s (consistent with
  proven latency-bound cross-node decode). EAGLE3 bf16-draft-vs-NVFP4-target dead-ends.
- Pages touched: models/minimax (TP=3 recipe, OOM fixes, NCCL shim, cold power-drain, EAGLE3
  status — all [conjecture]), multinode-tp-and-networking (LD_PRELOAD shim trap, cold
  power-drain bandwidth fix, bandwidth-vs-concurrency, Ray UMA false OOM — all [conjecture]),
  containers-and-tooling (NGC vs community container gap, nightly wheel pipeline, --vllm-ref,
  --name multi-container, VRAM soldered — all [conjecture]), models/laguna-s-2.1 (quality
  corroboration — good for reasoning+tools, fails generative tasks [conjecture]), benchmarks
  (Solar-Open2-250B INT4 on 2× Spark ~15 tok/s [conjecture]), sources/README, index, log.
- Skipped: 377689 (community extinction — social), 377733 (RMA prep), 374727 (entitlement).

## Forum ingest 2026-07-24 (Batch 33)
- 2 new forum topics found (1 technically relevant, 1 skipped as social/meta).
- 1 new source registered (Batch 33). 2 topic IDs added to processed_topics.txt (total now 442).
- **Headline finding:** Qwen3-TTS GGML backend crashes on GB10 — `ggml_cuda_kernel_can_use_pdl`
  `CUDA error: unspecified launch failure` on first inference. Root cause: qwentts-cpp-python
  PyPI wheels built against CUDA 12.8 / sm_120, not sm_121a. Memory ops work, compute kernels
  fail on dispatch. PDL error is a red herring (async CUDA errors are sticky). Fix: force
  `torch` backend (drop `[ggml]` extra, set `--qwen3_tts_backend torch`); CUDA graphs work,
  TTFA 2.65s, steady-state RTF ~1.7. Same sm_120-vs-sm_121a mismatch class as vLLM FP4 CUTLASS
  and Triton ptxas — now in a 3rd ecosystem component (GGML/qwentts.cpp). All [conjecture].
- Pages touched: platform-gb10 (GGML PDL crash root cause + sm_121a targeting gap
  [conjecture]), containers-and-tooling (Qwen3-TTS torch backend workaround + UMA audio
  tensor pinning tip [conjecture]), sources/README, index, log.
- Skipped: 377793 (PSA about using Discourse MCP to follow the forum — social/meta; replies
  reference already-sourced DSV4-Flash-DSpark and Laguna-S-2.1 recipes, no new findings).

## Forum ingest 2026-07-25 (Batch 34)
- 5 new forum topics found (2 technically relevant, 3 skipped: SSH config parser bug,
  macOS SSH tunnel manager, vision model recommendation thread).
- 2 new sources registered (Batch 34). 5 topic IDs added to processed_topics.txt (total now 447).
- Pages touched: containers-and-tooling (stock `vllm/vllm-openai:latest` hangs silently on GB10
  — no SM121 support; LocateAnything-3B bring-up — ARM64 wheel gaps, device_map='auto' UMA
  pitfall, FastAPI server pattern for non-vLLM models), platform-gb10 (device_map='auto' slow
  on 128 GB UMA), models/qwen (stock vLLM hang on Qwen3.6-35B-A3B-NVFP4). All [conjecture].
- Skipped: 378009 (NVIDIA Sync SSH config parser — client-side tooling), 377913 (macOS SSH
  tunnel manager — personal tool), 377759 (vision model recommendations — no durable findings).

## Forum ingest 2026-07-26 (Batch 36)
- 4 new forum topics found (3 technically relevant, 1 skipped as non-technical question with no
  answers/findings).
- 3 new sources registered (Batch 36). 4 topic IDs added to processed_topics.txt (total now 451).
- Pages touched: benchmarks (Solar-Open2-250B NVFP4 W4A4 on 2× Spark — 15.8 tok/s decode, flat
  with depth; FP8 KV = capacity lever not speed lever; full recipe + flags — [conjecture]),
  attention-and-kv-cache (hybrid-linear attention dodges KV-bandwidth wall; FP8 KV capacity vs
  speed distinction — [conjecture]), platform-gb10 (USB-C PD firmware pending update causes
  overheating without load; 30-min power-cycle fix — [conjecture]), engines (GLM-5.2-Vision-NVFP4
  frozen-backbone projector; adaptive MTP dynamic 2–5 draft depth — [conjecture]),
  roadmap (1 new open problem: adaptive MTP feedback-loop overhead on bandwidth-bound decode).
- Skipped: 378102 (CUDA_VISIBLE_DEVICES simulation question — no answers, no findings, buying
  advice context).

## Forum ingest 2026-07-27 (Batch 37)
- 3 new forum topics found (2 technically relevant, 1 skipped as application showcase).
- 2 new sources registered (Batch 37). 3 topic IDs added to processed_topics.txt (total now 462).
- **Headline finding:** Qwen3.5-122B-A10B-int4 is the community consensus single-Spark daily
  driver — 4 independent forum users confirm it as the "king model" → **[reported]**. New tok/s
  numbers: AutoRound int4 ~65 tok/s on 2× Spark (holds linearly past 100K context), FP8 ~35
  tok/s on 1× Spark, sparkrun-recipes patched vLLM v26 build 5 lanes @ 256K ctx 40+ tok/s.
  AutoRound int4 loop tendency flagged; NVFP4 variants may offer better fidelity.
- Pages touched: models/qwen (122B "king model" consensus + sparkrun-recipes + AutoRound loop),
  containers-and-tooling (sparkctl config-driven serving CLI), benchmarks (4 new [conjecture]
  rows: Qwen 122B int4/fp8/hybrid, DSV4-Flash 1× 45-50 tok/s), sources/README, index, log.
- Skipped: 378131 (brewing agent application showcase — Mistral 119B on Spark, no durable
  technical findings).

## Forum ingest 2026-07-27 (Batch 38)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 38). 2 topic IDs added to `sources/processed_topics.txt`
  (total now 464).
- **Headline finding:** GLM-5.2 hybrid FP8+NVFP4+MXFP4 — first reported 3-way mixed-precision
  checkpoint on GB10 (aidendle94). Decode ~20-25 tok/s on 4× Spark (same bandwidth-bound range
  as pure AWQ-INT4 / NVFP4). Tool-eval-bench 86/100 (v2) / 85/100 (v3-GPTQ). Structured Output
  58% root-caused to reasoning-parser bug (thinking off → 100%, +8 pts overall). Word-salad at
  >90k ctx root-caused to hardcoded `repetition_penalty=1.2` (config, not model/hardware). b12x
  sparse-MLA kernel only reads packed fp8 KV pages. New model page created: `wiki/models/glm-5.2.md`.
- Pages touched: models/glm-5.2 (NEW — consolidated all GLM-5.2 findings across 12 sources),
  benchmarks (4 new [conjecture] rows: hybrid v2/v3/llama-benchy-table/official-NVFP4),
  platform-gb10 (ASUS GX10 SoC+TPM firmware — stable, 2-4% noise, slow reboot),
  quantization-on-gb10 (3-way hybrid quant, custom NVFP4 KV cache, repetition_penalty sensitivity),
  engines (reasoning-parser structured-output bug, thinking-off A/B, MTP4 vs MTP5, word-salad root
  cause), attention-and-kv-cache (b12x sparse-MLA KV format constraint — bf16 KV → immediate EOS),
  sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture] (single thread, multiple
  users in same thread using same image → not independent).

## Forum ingest 2026-07-28 (Batch 39)
- 5 new forum topics found (4 technically relevant, 1 skipped: social/buying advice).
- 4 new sources registered (Batch 39). 5 topic IDs added to `sources/processed_topics.txt`
  (total now 469).
- **Headline finding:** Qwen 122B vLLM v26 + fp8 KV + DFlash + int8 lm-head on single Spark
  (styles01) — first working fp8 KV + DFlash on GB10 for hybrid quant models. 3 custom patches
  (inc_hybrid, int8_lmhead_v3, prefix_align). KV 549K→1.37M tokens (2.6×), concurrency 5.24×
  @ 256K, decode 45.98 tok/s, prefill 957 tok/s (+32%). int8 lm-head reclaims ~1.4 GB.
- Pages touched: models/qwen (v26 + fp8 KV + DFlash + int8 lm-head recipe + benchmark table),
  engines (SpeedyColibri — Rust port of Colibri for GLM-5.2, ~1→4 tok/s with fp8),
  containers-and-tooling (whisper.cpp STT Docker — 7 GB10 build gotchas; official llama.cpp
  Docker matches custom builds, --mmap 0 mandatory, power-cycle fixes 40→67 tok/s),
  benchmarks (4 new [conjecture] rows), sources/README, index, log.
- Skipped: 377281 (social/buying advice, 104 posts, no durable findings).
- No evidence promotions past [reported]. All new findings [conjecture].

## Forum ingest 2026-07-29 (Batch 40)
- 5 new forum topics found (3 technically relevant, 2 skipped: boot failure/RMA, power adapter buying advice).
- 3 new sources registered (Batch 40). 5 topic IDs added to `sources/processed_topics.txt`
  (total now 474).
- **Headline finding:** ComfyUI hard-crash root cause on GB10 — GPU power spike (14→85 W
  instantaneous) trips overcurrent protection, distinct from the power-controller wedge. Fix:
  `nvidia-smi -lgc 300,2100` (clock cap to 2100 MHz, ~50 W) + `swapoff -a`. Second user
  confirms clock cap stabilizes. `--highvram` is a trap on UMA; async offload is near-free.
  `CUDA_CACHE_MAXSIZE=4GB` gives 3× rerun speedup.
- **Gemma-4-26B-A4B NVFP4 benchmark:** Unsloth ~17% faster than nvidia on Spark (160 vs 128
  tok/s aggregate @100 concurrent). Spark ~6-7× slower than RTX Blackwell 6000 Pro. Corroborates
  Unsloth-vs-nvidia quant difference pattern (opposite direction from Qwen3.6-35B where Unsloth
  was slower).
- **CUDA MPS on Spark:** first documented MPS setup for multiple vLLM instances on one GB10.
  `EXCLUSIVE_PROCESS` mode + MPS daemon + `--gpu-memory-utilization 0.45` per instance. Latency
  increases, throughput modestly improves; main value is multi-model co-residency.
- Pages touched: platform-gb10 (power spike/overcurrent, clock cap, swapoff mechanism, CUDA_CACHE,
  --highvram UMA trap, ComfyUI no multi-GPU), models/gemma-4 (Unsloth vs nvidia NVFP4 benchmark),
  containers-and-tooling (ComfyUI crash fix recipe, CUDA MPS setup), benchmarks (2 new [conjecture]
  rows: gemma-4-26B unsloth/nvidia NVFP4), sources/README, index, log.
- Skipped: 378157 (won't boot after update — RMA/boot failure, no durable technical findings),
  378245 (power adapter + network connectivity — buying advice, basic setup question).
- No evidence promotions past [reported]. All new findings [conjecture].

## Forum ingest 2026-07-29 (Batch 41)
- 5 new forum topics found, all technically relevant.
- 5 new sources registered (Batch 41). 5 topic IDs added to `sources/processed_topics.txt`
  (total now 479).
- **Headline finding:** Hard power-off under sustained GPU load at ~90W — detailed reproduction
  with stepped FP16 matmul and throttle bit logging. Unit dies before thermal protection engages
  (no throttle flag, GPU only 82°C), persists after full platform firmware update (SOCFW/EC/USBPD
  all current), clock cap 2200 MHz fixes it. CPU 92-97°C while GPU 78-83°C — GPU sensor looks
  normal. DCGM cannot stress GB10 (Skip for all targeted tests). NVIDIA confirms known issue.
  Clock-cap mitigation now corroborated by 4 independent threads → [reported] as standard GB10
  power/thermal mitigation.
- **Unsloth+b12x vs nvidia+Marlin:** Unsloth ~8% faster than nvidia at 100 concurrent on Spark
  (436 vs 404 tok/s agg) — reverses the prior [reported] 15% slower finding. The b12x backend
  (not the quant) appears to be the lever. Working flashinfer_b12x recipe documented
  (CUTE_DSL_ARCH=sm_121a, vllm>=0.25.0, flashinfer>=0.6.13).
- **NVFP4 KV cache:** 1.68× more capacity than FP8 on Spark (2.31M vs 1.37M tokens, Qwen3-4B on
  SGLang). dtype `torch.float4_e2m1fn_x2`. Extends the KV quant ladder: bf16 → fp8 (2×) → NVFP4
  (1.68× over fp8).
- **DSV4-Flash REAP25 PrismaAURA:** third independent ds4 fork for single GB10 — 92/100 tool-eval,
  16.5 tok/s spec decode, measured-KL quant allocation (IQ2+MXFP4+MXFP8 mix via knapsack). Key
  GB10 finding: sub-4-bit formats (IQ2) cannot use tensor cores — dequant overhead negates compute
  advantage. MXFP4 is the only format that escapes to tensor cores natively. marco.palaferri fork
  achieves 854 tok/s prefill via HMMA attention. DSV4-Flash prefill is compute-bound, not
  bandwidth-bound — distinct from decode.
- Pages touched: platform-gb10 (hard power-off at 90W, clock cap [reported], DCGM Skip, GPU
  throttle commands), models/qwen (Unsloth+b12x benchmark, flashinfer_b12x recipe, vLLM 0.25.x
  hang), attention-and-kv-cache (NVFP4 KV cache capacity), quantization-on-gb10 (NVFP4 KV cache,
  sub-4-bit tensor-core wall, W4A8 source-faithful path), engines (DSV4 REAP25 measured-quant,
  marco.palaferri fork, IQ2 tensor-core wall, prefill compute-bound), benchmarks (8 new
  [conjecture] rows), sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture] except the clock-cap
  mitigation which reaches [reported] via 4 independent corroborating sources.

## Forum ingest 2026-07-30 (Batch 42)
- 4 new forum topics found (3 technically relevant, 1 skipped: RMA complaint).
- 3 new sources registered (Batch 42). 4 topic IDs added to `sources/processed_topics.txt`
  (total now 483).
- **Headline finding:** apt upgrade to driver 580.173.02 breaks GPU on OTA2607 —
  "torn" driver/firmware pairing. Ubuntu noble-updates serves 580.173.02 which
  is not paired with OTA2607's GSP/SEC2 firmware (expects 580.159.03). Xid 119,
  GSP_INIT_DONE timeout, nvidia-smi "No devices found". nvidia-spark-ota-check
  reports torn=1. Fix: re-run DGX Dashboard update or downgrade + hold driver.
  580.173.02 works on Sparks with matching firmware — failure is firmware-version-
  dependent, not universal.
- Pages touched: platform-gb10 (driver 580.173.02 torn pairing, USB3→USB2 fallback
  corroborated on Asus GX10 [reported, 8th user/4th OEM SKU], USB SSD intermittent
  20 MB/s drops, Acer Veriton GN100 thermal A/B ~68°C vs 80-82°C other OEMs,
  spark_hwmon power telemetry driver), containers-and-tooling (model storage
  strategies — 4TB NVMe, NFS 10GbE, NVMe-oF 400G, cron offloading, modelctl,
  USB2-at-boot gotcha), benchmarks (Acer thermal A/B table), sources/README,
  index, log.
- Skipped: 378356 (RTL8127 NIC defect — RMA complaint, 1 of 4 identical units
  affected, hardware fault not platform-wide).
- No evidence promotions past [reported]. All new findings [conjecture].

## Forum ingest 2026-07-30 (Batch 43)
- 5 new forum topics found (3 technically relevant, 2 skipped: karaoke app showcase,
  switch buying advice).
- 2 new sources registered (Batch 43). 5 topic IDs added to processed_topics.txt
  (total now 488).
- **Headline finding:** SM121 software support thread (357663, 43 posts) — NVIDIA
  official roadmap response + community fact-check. vLLM --enforce-eager 20-30% perf
  loss, CuTE DSL FP4 restricted to sm_100a (Issue #2800), PyTorch 2.10/Triton 3.6.0/
  FlashInfer 0.5.3+/CUTLASS 4.2.0+ roadmap, SGLang unofficial branch, MoE kernels
  no optimized GB10 configs, tcgen05/DSMEM/TMEM/TMA lacking, CUDA 12.0f vs 12.1a
  distinction, no locked/hidden memory on Spark.
- Pages touched: platform-gb10 (SM121 software support — 10 new [conjecture] findings),
  models/laguna-s-2.1 (DFlash acceptance corroborated by 5 independent users → [reported],
  tool-eval 82-87/100, TP=2 KV cache 2.7M tokens), roadmap (2 new open problems: CuTE
  DSL FP4 sm_100a restriction, vLLM 0.14.0 enforce-eager question), sources/README,
  index, log.
- Skipped: 378524 (AIraoke — app showcase, no GB10 findings), 378255 (switch buying
  advice — CRS504/CRS812 already documented).
- No evidence promotions past [reported].

## Forum ingest 2026-07-31 (Batch 44)
- 3 new forum topics found (1 technically relevant, 2 skipped: social intro, non-technical
  question with no answers).
- 1 new source registered (Batch 44). 3 topic IDs added to `sources/processed_topics.txt`
  (total now 491).
- **Finding:** Xid 31 MMU faults during AMP-enabled YOLOv8s training on GB10 — 5/5 AMP
  runs fault with `ENGINE GRAPHICS GPC2` / `FAULT_PDE ACCESS_TYPE_VIRT_READ`; cuDNN
  `CUDNN_STATUS_EXECUTION_FAILED` at conv2d; FP16 matmul and AMP-disabled training run
  clean. Root cause unresolved (cuDNN/driver/GSP/hardware all open). Non-LLM workload but
  documents a GB10-specific GPU fault signature under AMP conv paths. [conjecture].
- Pages touched: platform-gb10 (Xid 31 MMU fault), sources/README, log.
- Skipped: 378532 (social intro — replies reference already-sourced tools), 377567
  (non-technical question, no answers).
- No evidence promotions past [reported]. No new wiki pages created.
