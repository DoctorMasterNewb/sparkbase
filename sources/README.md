# Sources registry

Where wiki findings came from. Cite these `S-` ids in page `sources:` headers. Every source has a
**type** — it caps the evidence tier the source can justify on its own (see
[`../SCHEMA.md`](../SCHEMA.md) → evidence ladder & source types):

- `forum` / `repo` / `report` → `[conjecture]` (→ `[reported]` if independent sources agree)
- `first-party` → `[reproduced]` / `[proven]` (an experiment/bring-up run on a real DGX Spark)

`first-party` sources are cited by **what was run and when**, not by any private filesystem path,
hostname, or IP — this is a public repo (see SCHEMA → Sanitization).

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-xnode-cudagraph | first-party | Cross-node cudagraph capture fails on GB10 (reproduced + filed upstream) | vllm-project/vllm#46253 | 2026-06-20 |
| S-pr46372 | first-party | Upstream fix "make NCCL collectives eager break points in breakable cudagraph" (#46253) + staged GB10 test | vllm-project/vllm#46372 | 2026-06-29 |
| S-m3-vision | first-party | MiniMax-M3 vision + long-ctx cross-node bring-up | first-party bring-up, 2×GB10 cross-node | 2026-06-18 |
| S-m3-20tps | first-party | MiniMax-M3 "20 tok/s" throughput mission (cudagraph/MTP/PP walls) | first-party mission, 2×GB10 | 2026-06-20 |
| S-m3-eagle3 | first-party | MiniMax-M3 EAGLE3 spec-decode bring-up (draft-model card + overnight runs/benches) | first-party runs + HF `Inferact/MiniMax-M3-EAGLE3` draft card | 2026-07-03 |
| S-minimax-sweeps | first-party | MiniMax-M2.7 AWQ vs NVFP4 benchmark sweeps (pp/tg × concurrency, EP=2) | first-party llama-benchy sweeps, single-node | 2026-06-09 |
| S-mimo-results | first-party | MiMo-V2.5-NVFP4 bring-up results (mods chain, abliteration verdict) | first-party bring-up | 2026-06-21 |
| S-mimo-doc | first-party | MiMo-V2.5-NVFP4 runtime notes (mods rationale, startup markers) | first-party runtime doc | 2026-06 |
| S-dflash-nvfp4 | first-party | DFlash speculative decoding on MiMo-V2.5-NVFP4 (custom proposer, nvfp4-at-depth, full-context) | first-party bring-up | 2026-07-06 |
| S-networking | first-party | DGX Spark networking (CX7 twins, NCCL, netplan, NCCL tests) | first-party fabric bring-up notes | 2026-06 |
| S-nemotron-rpc | first-party | Nemotron-3-Super-120B Q8 llama.cpp RPC 2-node trial | first-party trial, 2×GB10 | 2026-06-15 |
| S-diffusiongemma | first-party | DiffusionGemma-26B-A4B bring-up (native-support probe, NVFP4 marlin-force, bf16-vs-NVFP4, retirement) | first-party bring-up + HF `google/diffusiongemma-26B-A4B-it` | 2026-07-01 |
| S-swapper-sweep | first-party | Full serving-menu benchmark sweep (single methodology, post power-cycle + cross-node fixes) | first-party sweep, 2×GB10 | 2026-06-30 |
| S-swapper | first-party | Serving-supervisor conventions (single-port, alias, swap-on-conflict) | first-party ops conventions | 2026-06-28 |
| S-memory | first-party | Distilled bring-up notes (networking, vllm-nvfp4, atlas, holo) | first-party accumulated notes | ongoing |
| S-spark-powercap | report | GPU power-controller wedge ("14 W cap", pinned clock) + AC power-cycle fix — external writeup, first-party corroborated | dredyson.com (DGX Spark performance-degradation writeup); first-party `nvidia-smi` measurement | 2026-06-30 |
| S-dgxspark-report | report | External deep-research report "Optimizing LLM Inference on the NVIDIA DGX Spark GB10" (SoC specs, fabric caveats, model benchmarks) | third-party report (cites TechPowerUp, LMSYS DGX-Spark review, NVIDIA marketplace) | 2026-07-01 |

## First-party session notes

Several early findings were distilled from first-party Claude Code bring-up sessions on the reference
2×DGX-Spark. They're cited generically (the raw transcripts are private, not part of this repo):

| ID | What | Date |
|---|---|---|
| S-sess-jun5 | Platform/memory-bandwidth characterization notes | ~Jun 5 |
| S-sess-jun11 | Multi-node TP / networking bring-up notes | ~Jun 11 |
| S-sess-early | Early platform + engine notes | late May–Jun 1 |

## Forum sources (2026-07-08 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-fp4psa | forum | PSA: State of FP4/NVFP4 Support — AWQ 4-bit outperforms NVFP4 decode ~32% on GB10 (eugr) | https://forums.developer.nvidia.com/t/353069 | 2025-12-01 |
| S-forum-mxfp4-patches | forum | vLLM 0.17.0 MXFP4 patches for DGX Spark: BF16→MXFP4 online quant, SM121 CUTLASS fixes (amasawa_seiji) | https://forums.developer.nvidia.com/t/362824 | 2026-03-09 |
| S-forum-nvfp4-ray | forum | Help: NVFP4 model on 2x Spark with vLLM+Ray — CUTLASS FP4 fails on sm_121, ptxas 12.8 lacks sm_121a | https://forums.developer.nvidia.com/t/353723 | 2025-12-06 |
| S-forum-nvfp4-100b | forum | NVFP4 quantization of 100B-class Llama on 2x Spark — modelopt distributed quant pipeline, 6-fix list (kai.koehler) | https://forums.developer.nvidia.com/t/370068 | 2026-05-13 |
| S-forum-qwen122 | forum | Qwen3.5-122B-A10B on single Spark up to 51 tok/s — patches + quick-start (eugr) | https://forums.developer.nvidia.com/t/365639 | 2026-02-02 |
| S-forum-dsv4-flash | forum | DeepSeek-V4-Flash official FP8 on 2x Spark TP=2 — ~44 tok/s decode, MTP, 200K ctx recipe (tonyd615) | https://forums.developer.nvidia.com/t/370309 | 2026-05-16 |
| S-forum-dsv4-dspark | forum | DeepSeek-V4-Flash-DSpark on 2x Spark — ~60-67 tok/s code, 1M ctx, NVFP4 KV, concurrency (tonyd2wild) | https://forums.developer.nvidia.com/t/374846 | 2026-06-29 |
| S-forum-glm52-4x | forum | GLM-5.2 on 4x GB10: ~22 tok/s decode, 256K ctx, AWQ-INT4 + 15% expert prune + MTP (CosmicRaisins) | https://forums.developer.nvidia.com/t/374125 | 2026-06-22 |
| S-forum-mimo-2x | forum | MiMo-V2.5-NVFP4 on 2x Spark — recipe, findings, fixes, benchmarks (a3refaat) | https://forums.developer.nvidia.com/t/370459 | 2026-05-18 |
| S-forum-mimo-3x | forum | MiMo V2.5 Omni on 3x Spark TP=3 + MTP + 1M ctx 39 tok/s — virtual-head padding (tonyd615) | https://forums.developer.nvidia.com/t/373948 | 2026-06-20 |
| S-forum-mimo-tp2-1m | forum | MiMo-V2.5 Omni TP=2 1M context NVFP4 KV on 2x Spark — ~30 tok/s (tonyd2wild) | https://forums.developer.nvidia.com/t/374262 | 2026-06-23 |
| S-forum-m3-nvfp4-4x | forum | MiniMax-M3-NVFP4 on 4x Spark — chthonic vLLM, b12x backend, FULL cudagraph, 524K ctx (OllieJW) | https://forums.developer.nvidia.com/t/373927 | 2026-06-20 |
| S-forum-m3-awq-4x | forum | MiniMax-M3-AWQ on 4x GB10 fp8 KV 262k ctx ~30 tok/s (Sebesky) | https://forums.developer.nvidia.com/t/374175 | 2026-06-26 |
| S-forum-m3-llamacpp-2x | forum | MiniMax-M3 426B on 2 nodes via llama.cpp RPC — ~10.7 tok/s UD-IQ4_XS, hybrid tool template (karol.spark) | https://forums.developer.nvidia.com/t/373421 | 2026-06-15 |
| S-forum-clock721 | forum | GB10 GPU clock pinned at 721 MHz under load, ~10 W, no throttle flag — AC power-cycle fix (vaclav.sisl) | https://forums.developer.nvidia.com/t/376039 | 2026-07-08 |
| S-forum-power-crash | forum | GB10 power limited after crash — 5-9 W, fixed only by unplugging power brick (moreleatherjackets) | https://forums.developer.nvidia.com/t/366590 | 2026-04-14 |
| S-forum-15w-loop | forum | GPU trapped in 15W/650MHz loop with 50°C artificial T.Limit — SW power cap, AC power-cycle fix (nilayparikh) | https://forums.developer.nvidia.com/t/370304 | 2026-05-16 |
| S-forum-60w-cap | forum | GB10 Asus GX10 maxing at 60W — SW power capping counter, CPU+GPU share 140W envelope (thedivrox) | https://forums.developer.nvidia.com/t/374791 | 2026-06-28 |
| S-forum-cx7-13gbps | forum | ConnectX-7 inter-Spark link capped at ~13 Gbps — PCIe SlotPowerLimit 0W causes driver throttle (Ank-Chy) | https://forums.developer.nvidia.com/t/363461 | 2026-03-14 |
| S-forum-m3-quad | forum | MiniMax M3 NVFP4 for Quad DGX Spark — MSA architecture overview, 1M context (eh17) | https://forums.developer.nvidia.com/t/372123 | 2026-06-03 |

## Batch 2 forum sources (2026-07-08)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-atlas | forum | Atlas engine: Rust+CUDA, 82 tok/s Qwen3-Next-80B on 1x Spark, 2.8x vLLM (tbraun96) | https://forums.developer.nvidia.com/t/362210 | 2026-03-02 |
| S-forum-dflash-qwen122 | forum | DFlash for Qwen3.5-122B-A10B: 80+ tok/s on 1x Spark via block-spec decode (entrpi) | https://forums.developer.nvidia.com/t/374328 | 2026-06-24 |
| S-forum-hy3 | forum | Hy3-295B NVFP4-W4A16 + MTP on 2x Spark: 21.8 tok/s, enforce-eager wins (tonyd615) | https://forums.developer.nvidia.com/t/375851 | 2026-07-07 |
| S-forum-ds4-cuda | forum | antirez/ds4 (DwarfStar 4) custom CUDA-native DS4F on 1x Spark: ~28 tok/s Q2 (entrpi) | https://forums.developer.nvidia.com/t/369791 | 2026-05-12 |
| S-forum-glm52-mtp-fix | forum | GLM-5.2 NVFP4 4x: MTP vLLM config bug fixed, 24 tok/s at 128K ctx MTP4 (mattw-kz) | https://forums.developer.nvidia.com/t/375416 | 2026-07-03 |
| S-forum-qwen27-nvfp4 | forum | Qwen3.6-27B FP16 vs NVFP4 MMLU: 0.8446 vs 0.8485 — NVFP4 quality preserved (shahizat) | https://forums.developer.nvidia.com/t/375094 | 2026-07-01 |
| S-forum-m3-mxfp4-4x | forum | MiniMax-M3-MXFP4 4x TP=4 EAGLE3: ~35 tok/s, fp8 KV crashes on startup (bokunogf) | https://forums.developer.nvidia.com/t/375386 | 2026-07-03 |
| S-forum-m3-awq-1m | forum | MiniMax-M3-AWQ 1M ctx 4x: nvfp4 KV 25 tok/s, inline-dequant kernel fused (tonyd615) | https://forums.developer.nvidia.com/t/375372 | 2026-07-02 |
| S-forum-m3-awq-tp4 | forum | MiniMax-M3-AWQ TP=4 4x: 33 tok/s, CUDA 13.0 mismatch fix, 5 GB10 build fixes (tonyd615) | https://forums.developer.nvidia.com/t/375361 | 2026-07-02 |
| S-forum-m3-vision-b12x | forum | Vision on M3-W4A16-GPTQ b12x: 33 tok/s + OCR multimodal reproduced on 2x (tonyd615) | https://forums.developer.nvidia.com/t/375687 | 2026-07-06 |
| S-forum-m3-eagle3-2x | forum | MiniMax-M3 EAGLE3 on 2x: ~14-15 tok/s, better quality than M2.7 (tonyd615) | https://forums.developer.nvidia.com/t/375475 | 2026-07-04 |
| S-forum-glm52-1bit | forum | GLM-5.2 1-bit UD-IQ1_S RPC llama.cpp 2x: 8 tok/s, 256K ctx, toy experiment | https://forums.developer.nvidia.com/t/374523 | 2026-06-26 |
| S-forum-glm52-reapless | forum | REAP-less GLM-5.2 NVFP4 on 4x Spark: 128K ctx, >22 tps DCP=1 | https://forums.developer.nvidia.com/t/374832 | 2026-06-30 |
| S-forum-glm52-800k | forum | GLM-5.2 unpruned 800K ctx 4x: 20-33 tok/s NF3-hybrid checkpoint | https://forums.developer.nvidia.com/t/375909 | 2026-07-07 |
| S-forum-tma | forum | TMA not on GB10: consumer Blackwell lacks TMEM (datacenter-only) (s0ne) | https://forums.developer.nvidia.com/t/374243 | 2026-06-23 |
| S-forum-driver610 | forum | Driver 610.43.02 + CUDA 13.3 on Spark: 82W under vLLM, needs secureboot disable | https://forums.developer.nvidia.com/t/373994 | 2026-06-21 |
| S-forum-ubuntu2604 | forum | Ubuntu 26.04 + drivers 610 + CUDA 13.3 + ZFS on GX10: CX7 power fix, clean install | https://forums.developer.nvidia.com/t/373655 | 2026-06-17 |
| S-forum-power-spec | report | NVIDIA official: 240W total system, 140W GB10 SoC TDP, 100W rest (CX7+SSD+USB) | https://forums.developer.nvidia.com/t/349668 | 2025-10-31 |
| S-forum-mikrotik | forum | MikroTik CRS812/CRS504 for 4x Spark: breakout cables, 100G/200G options | https://forums.developer.nvidia.com/t/373273 | 2026-06-13 |
| S-forum-ddp-timeout | forum | NCCL ALLREDUCE timeout during DDP training on 2x Spark: rank desync | https://forums.developer.nvidia.com/t/366147 | 2026-04-09 |
| S-forum-2d-parallel | forum | 4x Spark 2D parallelism (TPxPP): PP over RJ-45 too latency-sensitive (eugr_nv) | https://forums.developer.nvidia.com/t/375837 | 2026-07-07 |
| S-forum-kernel-panic | forum | Kernel panic after dashboard update: initramfs missing, GRUB_TIMEOUT=0 blocks recovery | https://forums.developer.nvidia.com/t/368939 | 2026-05-04 |
| S-forum-gsp-timeout | forum | GPU GSP_INIT_DONE timeout (Xid 119) + SEC2 secure-boot timeout after OTA firmware | https://forums.developer.nvidia.com/t/373394 | 2026-06-15 |
| S-forum-nemotron-4x | forum | Nemotron-3-Ultra on 4 DGX Sparks: deployment guide by eugr for NVIDIA | https://forums.developer.nvidia.com/t/374282 | 2026-06-21 |
| S-forum-thermal | forum | DGX Spark overheating shutdowns during ComfyUI: specific units, RMA recommended | https://forums.developer.nvidia.com/t/363370 | 2026-03-13 |
| S-forum-cooling-cage | forum | Dual Spark ducted cooling cage: Noctua 120mm, 3D-printed, idle 40C GPU | https://forums.developer.nvidia.com/t/365302 | 2026-04-01 |
| S-forum-headless-boot | forum | Auto-power-on for headless DGX Sparks: BIOS setting guide | https://forums.developer.nvidia.com/t/374176 | 2026-06-24 |
| S-forum-cve | forum | Security Bulletin CVE-2026-24218 for NVIDIA DGX Spark | https://forums.developer.nvidia.com/t/374930 | 2026-07-01 |
| S-forum-tool-eval | forum | Tool Eval Bench CLI: benchmark tool for DGX Spark model evaluation | https://forums.developer.nvidia.com/t/366903 | 2026-04-14 |
| S-forum-nemotron-sm121 | forum | Nemotron-3-Super 120B on GB10: llama.cpp sm_121 build + Ollama GGUF fix | https://forums.developer.nvidia.com/t/363459 | 2026-03-10 |
| S-forum-ds4f-single | forum | DeepSeek V4 Flash IQ2XXS on single GB10: fits in 128GB, runs | https://forums.developer.nvidia.com/t/368970 | 2026-05-07 |
| S-forum-qwen397-1m | forum | Qwen3.5-397B-A17B at 1M tokens on 2x Spark: AutoRound int4 | https://forums.developer.nvidia.com/t/375421 | 2026-07-04 |
| S-forum-ornith | forum | Ornith-1.0-397B/35B: self-scaffolding LLMs for agentic coding, int4-AutoRound | https://forums.developer.nvidia.com/t/374601 | 2026-06-28 |
| S-forum-ornith-int4 | forum | Ornith-1.0-35B-int4-AutoRound for GB10: community quant | https://forums.developer.nvidia.com/t/374801 | 2026-06-30 |
| S-forum-gigachat | forum | GigaChat3.5-432B-A28B: new large MoE, quants may fit 2x Sparks | https://forums.developer.nvidia.com/t/375814 | 2026-07-06 |
| S-forum-prismaquant | forum | PrismaScout/PrismaQuant v2: community quantization tool for GB10 MoE | https://forums.developer.nvidia.com/t/368933 | 2026-04-25 |
| S-forum-mimo-pro-dflash | forum | MiMo-V2.5-Pro-FP4-DFlash: 1000 tok/s on 1T model (Xiaomi/TileRT) | https://forums.developer.nvidia.com/t/372652 | 2026-06-13 |
| S-forum-mimo-dflash-30 | forum | MiMo V2.5 Dflash FP8 KV 1.5M+ context up to 30 tk/s on 2x Spark | https://forums.developer.nvidia.com/t/375945 | 2026-07-07 |
| S-forum-m3-reap | forum | MiniMax M3 NVFP4 and REAP-50 experimental sparkrun recipes (eugr) | https://forums.developer.nvidia.com/t/373177 | 2026-06-05 |
| S-forum-hermes-twin | forum | Asus GX10 stable: llama.cpp 3 instances (Qwen3.6+Hermes+ComfyUI) | https://forums.developer.nvidia.com/t/373094 | 2026-06-17 |
| S-forum-qwen-agentworld | forum | Qwen Agent World 35B-A3B for local programming: NVFP4 quant | https://forums.developer.nvidia.com/t/374835 | 2026-06-30 |
| S-forum-ds4f-guide | forum | DeepSeek-V4-Flash reproducible vLLM guide 2x Spark up to 1M context | https://forums.developer.nvidia.com/t/374742 | 2026-06-30 |
| S-forum-vllm-claude | forum | Docker image: NVIDIA vLLM 0.23.0 with Claude Code 2.1.195+ compatibility | https://forums.developer.nvidia.com/t/374827 | 2026-06-30 |
| S-forum-btop | forum | btop for DGX Spark: modified fork showing GPU resource info on GB10 | https://forums.developer.nvidia.com/t/356729 | 2026-02-25 |
| S-forum-model-manager | forum | DGX Spark Model Manager: open-source web UI for Ollama, SGLang & LiteLLM | https://forums.developer.nvidia.com/t/365394 | 2026-03-28 |
| S-forum-sparkdash | forum | sparkdash: monitoring/control dashboard for sparkrun DGX Spark clusters | https://forums.developer.nvidia.com/t/375391 | 2026-07-05 |
| S-forum-thunderkittens | forum | ThunderKittens 2.0: Blackwell support, tile primitives for speedy kernels | https://forums.developer.nvidia.com/t/361776 | 2026-03-06 |
| S-forum-dp-mst | forum | GB10 driver 580.159.03: dual DP-MST stream scanout fails (single works) | https://forums.developer.nvidia.com/t/372133 | 2026-06-20 |
| S-forum-xhci | forum | XHCI Controller HC Died crashes with RealSense D435i on DGX Spark | https://forums.developer.nvidia.com/t/355453 | 2026-03-10 |
| S-forum-wifi-mt7925 | forum | MT7925e WiFi cannot connect after OOBE: PTK key addition failed | https://forums.developer.nvidia.com/t/374231 | 2026-06-23 |
| S-forum-soft-lockup-dp | forum | Soft lockup in nvidia_modeset DisplayPort path during Xorg logout | https://forums.developer.nvidia.com/t/371009 | 2026-05-20 |

## Batch 3 forum sources (2026-07-09 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-mimo-dflash-v024 | forum | MiMo-V2.5 + DFlash + 4-bit NVFP4 KV cache in one vLLM v0.24.0 instance — custom_class drafter decoupled from global allocator (danielgbates) | https://forums.developer.nvidia.com/t/375923 | 2026-07-07 |
| S-forum-mimo-dflash-22-67 | forum | MiMo-V2.5 + DFlash spec-decode on 2× Spark: 22→67 tok/s depending on workload (acceptance scales with output structure) (danielgbates) | https://forums.developer.nvidia.com/t/375607 | 2026-07-05 |
| S-forum-m3-w4a16-gptq | forum | MiniMax-M3-W4A16-GPTQ 2×GB10 deployment: 36 tok/s, fp8/nvfp4/KVarN/EAGLE3, b12x + vllm (a3refaat) | https://forums.developer.nvidia.com/t/375595 | 2026-07-04 |
| S-forum-glm47-full-2x | forum | Full GLM-4.7 (355B) NVFP4 at 64K context on 2× Spark GB10 — 17.5 tok/s, TP=2, 4 walls documented (LeatheryTendons) | https://forums.developer.nvidia.com/t/375690 | 2026-07-06 |
| S-forum-vllm-2606-broken | forum | nvcr.io/nvidia/vllm:26.06-py3 HTTP 500 on every API request — prometheus-fastapi-instrumentator + fastapi 0.136+ incompat (bartlomiej.niton) | https://forums.developer.nvidia.com/t/375743 | 2026-07-06 |
| S-forum-qwen36-27b-recipe | forum | Best 2× Spark Qwen 3.6 27B recipe — NVFP4 Marlin, TP=2, 262K ctx, MTP (ivr718, jrsphd) | https://forums.developer.nvidia.com/t/375360 | 2026-07-02 |
| S-forum-qwen36-27b-fp8 | forum | Qwen 3.6 27B FP8 without quantizing: MTP nst=3 → 1.94× speedup (7.8→15.2 tok/s); bandwidth-bound ~10 tok/s ceiling (starkrun) | https://forums.developer.nvidia.com/t/367561 | 2026-04-23 |
| S-forum-gemma4-assistant | forum | Gemma4-31B-IT-NVFP4 + assistant MTP drafter via EUGR vLLM fork — 14.1 tok/s decode, MTP=7 optimal (eugr_nv, jwarner) | https://forums.developer.nvidia.com/t/370194 | 2026-05-14 |
| S-forum-device-hang | forum | DGX Spark hangs under load — OOM on unified memory, driver 580.159.03+ kills process instead (aniculescu) | https://forums.developer.nvidia.com/t/375016 | 2026-06-30 |
| S-forum-cx7-bricked | forum | ConnectX-7 bricked by unsolicited mlnx-fw-updater firmware flash — stuck in pre-init/static_config_not_done, error -110 (abrooksdavis) | https://forums.developer.nvidia.com/t/373900 | 2026-06-19 |
| S-forum-4node-mesh | forum | 4-node DGX Spark cluster without a switch — 200GBASE-SR4 transceivers + MPO breakout for full mesh, ~5W/node (mashie) | https://forums.developer.nvidia.com/t/368726 | 2026-05-01 |
| S-forum-spark-auto-round | forum | Spark Auto Round: Int4 AutoRound quant tool for GB10, OpenCode Instruct dataset, sensitivity-aware layer selection (whpthomas) | https://forums.developer.nvidia.com/t/373475 | 2026-06-16 |
| S-forum-ds4f-4x-vllm | forum | DeepSeek-V4-Flash on 4× Spark via vLLM jasl fork TP=4 RDMA MTP — 49–54 tok/s single-stream, NCCL 2.30.4 is the critical fix (Verel-lab) | https://forums.developer.nvidia.com/t/373808 | 2026-06-18 |
| S-forum-nemotron-super-mtp | forum | Nemotron-3-Super-120B MTP on 4× Spark via SGLang — 1.70× single-stream, accept_len ≈2.7, 3/4 depth beats NVIDIA cookbook 5/5 (ht12) | https://forums.developer.nvidia.com/t/373625 | 2026-06-17 |
| S-forum-sglang-traps | forum | SGLang multi-node on DGX Spark: 3 traps (false-positive collective mismatch, EAGLE flags on every node, RDMA passthrough) (Verel-lab) | https://forums.developer.nvidia.com/t/373677 | 2026-06-18 |
| S-forum-m25-sglang-4x | forum | MiniMax-M2.5-NVFP4 on 4× Spark via SGLang TP=4 EP=4: 124 tok/s agg@n8, CUTLASS MoE compile OOM fix MAX_JOBS=1 (Verel-lab) | https://forums.developer.nvidia.com/t/373676 | 2026-06-18 |
| S-forum-glm47-rdma | forum | GLM-4.7-FP8 on 4× Spark SGLang: 2.5× speedup (8.2→25 tok/s) just by enabling RDMA — SGLang container needs --device=/dev/infiniband (Verel-lab) | https://forums.developer.nvidia.com/t/373675 | 2026-06-18 |
| S-forum-kvarn | forum | KVarN: native vLLM KV-cache quantization backend by Huawei — 3-5× more KV capacity, calibration-free, one flag; Qwen 3.6 compatibility issue (adg1) | https://forums.developer.nvidia.com/t/372333 | 2026-06-04 |
| S-forum-nvmeof-expert | forum | NVMe-oF over ConnectX-7 for MoE expert streaming — unexplored path to >128GB models on single Spark, no GPUDirect needed (lvmnky) | https://forums.developer.nvidia.com/t/368358 | 2026-04-29 |
| S-forum-sdpa-corruption | forum | sm_121 silent SDPA EFFICIENT_ATTENTION corruption in custom PyTorch build — output norms 1.5×–27× off, MATH/FLASH correct (ht12) | https://forums.developer.nvidia.com/t/368005 | 2026-04-27 |
| S-forum-gb10-baseline | forum | GB10 Hardware Baseline: community probes for UMA latency/atomic/bandwidth — 161 GB/s idle, 90 GB/s under load (parallelArchitect) | https://forums.developer.nvidia.com/t/367851 | 2026-04-25 |
| S-forum-ddtree-dflash | forum | DDTree + DFlash: draft-tree method on block-diffusion, higher acceptance rates; 80+ tok/s Qwen3.5-27B AWQ claimed (joshua.dale.warner) | https://forums.developer.nvidia.com/t/366643 | 2026-04-15 |
| S-forum-roce-397b-mtp | forum | Two multi-node wins: RoCE 2× throughput + Qwen3.5-397B NVFP4 serving with SM121 CUTLASS patch; MTP +86% single-stream, 40 tok/s@n1 (ht12) | https://forums.developer.nvidia.com/t/366325 | 2026-04-12 |
| S-forum-sm121-kernel-guide | forum | DGX Spark 13→49 tok/s Qwen3.5-35B native SM121 kernel build guide — .so injection, CMake arch guard bug (troy.e.davis) | https://forums.developer.nvidia.com/t/365083 | 2026-03-30 |
| S-forum-glm52-iq4xs-4x | forum | GLM-5.2 IQ4_XS on 4× GB10 — 6.28 tok/s decode, DSA active, ngram self-spec →24 tok/s structured, full recipe (Mike_MK) | https://forums.developer.nvidia.com/t/373933 | 2026-06-20 |
| S-forum-nemotron-ultra-4x | forum | Nemotron-3-Ultra-550B-A55B-NVFP4 on 4× Spark SGLang TP=4 EP=4 RoCE — ~42-43 tok/s n8 peak, 512K ctx (ht12) | https://forums.developer.nvidia.com/t/372680 | 2026-06-09 |
| S-forum-gemma4-qat | forum | Gemma4 official QAT models incl W4A16 — google/gemma-4-31B-it-qat-w4a16-ct, MTP assistants work with QAT (jwarner) | https://forums.developer.nvidia.com/t/372444 | 2026-06-05 |
| S-forum-mistral-s4-nvfp4 | forum | Mistral-Small-4-119B-2603-NVFP4 — OOM during safetensors parse on 2× GB10, needs util 0.9 + swap (bugsareyummy) | https://forums.developer.nvidia.com/t/372427 | 2026-06-05 |
| S-forum-gemma4-mtp-4x | forum | Gemma-4-31B + MTP on 4× Spark SGLang — +154% @ n1 (26.68 tok/s), +80% @ n8 (153 tok/s), FROZEN_KV_MTP drafter (ht12) | https://forums.developer.nvidia.com/t/370354 | 2026-05-16 |
| S-forum-vllm-019-vs-023 | forum | vLLM 0.19 vs 0.23 regression: Qwen3.5-122B AutoRound 37→32 tok/s, memory 104→120 GB — performance + footprint regression (xkm121) | https://forums.developer.nvidia.com/t/375786 | 2026-07-06 |
| S-forum-nvfp4-quant-gp10 | forum | NVFP4 quantization on DGX Spark via TensorRT Model Optimizer — zero GPU load, CPU-bound, fails silently (nate.gelbard) | https://forums.developer.nvidia.com/t/348668 | 2025-10-22 |
| S-forum-ds4f-hybrid-1x | forum | DeepSeek-V4-Flash hybrid 2-bit quant on 1× Spark vLLM — antirez MLX recipe ported, ~85 GiB, coherent output (entrpi) | https://forums.developer.nvidia.com/t/369584 | 2026-05-10 |
| S-forum-kv-bench-llamacpp | forum | KV cache quantization benchmarks on Spark llama.cpp — q4_0 92% slower @ 64K, uses MORE memory than f16; q8_0 only worthwhile quant (nmaine) | https://forums.developer.nvidia.com/t/365138 | 2026-03-31 |
| S-forum-nemotron-super-abi | forum | Nemotron-3-Super NVFP4 vLLM TP=2 — 24 tok/s, ABI fix for cu130/cu132 mismatch in Dockerfile (leon-gibat) | https://forums.developer.nvidia.com/t/364862 | 2026-03-26 |
| S-forum-qwen122-nvfp4-redhat | forum | RedHatAI/Qwen3.5-122B-A10B-NVFP4 best option for single Spark — 16 tok/s, quality close to FP16 (gpieceoffice) | https://forums.developer.nvidia.com/t/363815 | 2026-03-17 |
| S-forum-qwen122-nvfp4-quant | forum | Qwen3.5-122B-A10B NVFP4 quantized 234GB→75GB, runs on 128GB — DeltaNet+vision, llm_head/routers kept BF16 (alper.tor) | https://forums.developer.nvidia.com/t/361819 | 2026-02-26 |
| S-forum-turboquant | forum | TurboQuant KV cache integration on vLLM 0.19.1: 155K→413K token capacity, gather-free Triton decode, CUDA WPH (bjk110) | https://forums.developer.nvidia.com/t/365627 | 2026-04-05 |
| S-forum-stream-loading | forum | vLLM custom for DGX Spark — STREAM LOADING (on-the-fly 4-bit quant), NF4 sub-mode, automatic KV cache allocation (amasawa_seiji) | https://forums.developer.nvidia.com/t/365798 | 2026-04-07 |
| S-forum-qwen35-35b-opt | forum | Qwen3.5-35B-A3B optimizations on single Spark — hybrid INT4+FP8 + MTP, 100+ tok/s, DFlash drafter (joshua.dale.warner) | https://forums.developer.nvidia.com/t/366326 | 2026-04-12 |
| S-forum-nvfp4-mistral-3node | forum | NVFP4 quant of 123B Mistral-Large finetune on 3-node heterogeneous cluster (2× Spark + 1× RTX 3090) via Ray (kai.koehler) | https://forums.developer.nvidia.com/t/370266 | 2026-05-15 |
| S-forum-flux2-nunchaku | forum | Flux.2 Klein 9B on DGX Spark: 2.5× faster inference, 59% lower VRAM with Vitoom Nunchaku quantized transformer+text encoder (tonera) | https://forums.developer.nvidia.com/t/374419 | 2026-06-25 |
| S-forum-sage-attn | forum | ComfyUI --use-sage-attention silently falls back to PyTorch attention (missing python3.12-dev → 20× slowdown); fix: apt install python3.12-dev (wentbackward) | https://forums.developer.nvidia.com/t/375830 | 2026-07-06 |
| S-forum-fwupd-mismatch | forum | DGX Dashboard Updates page hangs indefinitely — fwupd/libfwupd version mismatch after OTA 7.5.0, fwupd.service fails (thrashvtx) | https://forums.developer.nvidia.com/t/375537 | 2026-07-04 |
| S-forum-nvfp4-error-gp10 | forum | NVFP4 quantization on GP10 error — ModelOpt container CPU-bound, zero GPU load, multiple users report failure (nate.gelbard) | https://forums.developer.nvidia.com/t/348668 | 2025-10-22 |
| S-forum-qwen-tts-arm64 | forum | torchaudio installation failure on ARM64 — no ABI-compatible wheel for CUDA 13/SM 12.1, blocks Qwen3-TTS (ferdinando.tammaro) | https://forums.developer.nvidia.com/t/359663 | 2026-02-04 |
| S-forum-comfyui-container | forum | ComfyUI container for DGX Spark — ComfyUI-Nvidia-Docker with SageAttention, ONNX Runtime, uid/gid config (martial) | https://forums.developer.nvidia.com/t/363342 | 2026-03-13 |
| S-forum-llamacpp-container | forum | Building llama.cpp container images for Spark/GB10 — LD_LIBRARY_PATH fix for cuda-13/compat, CMAKE_CUDA_ARCHITECTURES=121a-real (cosinus) | https://forums.developer.nvidia.com/t/353664 | 2025-12-05 |
| S-forum-step37-llamacpp | forum | Step-3.7-Flash on single Spark via llama.cpp — IQ4_XS, 262K ctx, 31 tok/s decode, 11 tok/s @ max ctx (joshua.dale.warner) | https://forums.developer.nvidia.com/t/371804 | 2026-05-30 |

## Batch 4 forum sources (2026-07-10 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-flux2-nvfp4-compute | forum | FLUX.2-dev on Spark: ~3× faster with torchao NVFP4 W4A4 (activation-quantized) real FP4 compute via Triton, not weight-only; modelopt_fp4 hits diffusers unpack bug on sm_121a (vr8vr8) | https://forums.developer.nvidia.com/t/376106 | 2026-07-08 |
| S-forum-qwen35-lora-uma | forum | Bf16 LoRA fine-tuning of Qwen3.5-35B-A3B on DGX Spark — UMA mmap double-allocation OOM workaround (_EagerSafeOpen + posix_fadvise); FSDP from_pretrained full-model-per-rank (danielkreuzhofer, jesse75) | https://forums.developer.nvidia.com/t/363268 | 2026-03-12 |
| S-forum-llama-benchy | forum | llama-benchy: llama-bench-style benchmarking for ANY OpenAI-compatible endpoint — context-depth sweep, concurrency; demo numbers on dual + single Spark (eugr) | https://forums.developer.nvidia.com/t/356698 | 2026-01-06 |
| S-forum-opal-uefi | forum | TCG OPAL password + UEFI admin password corrupted after unexpected shutdown — firmware capsule updates locked out even post-reimage (cvella) | https://forums.developer.nvidia.com/t/368949 | 2026-05-04 |
| S-forum-wan2gp-onnx | forum | Wan2GP on DGX Spark: ONNX Runtime GPU device discovery fails on GB10 (/sys/class/drm/card0/device/vendor missing) — safely ignorable (kdb8756) | https://forums.developer.nvidia.com/t/353793 | 2025-12-07 |
| S-forum-sunshine-rdp | forum | Headless Sunshine remote desktop for DGX Spark — GB10 internal display controller 165 MHz max pixel clock, 4K@60 impossible, 1440p@120Hz best (mail.eelbaz, LsDmTandAI) | https://forums.developer.nvidia.com/t/348220 | 2025-10-19 |
| S-forum-cluster-dashboard | forum | DGX Spark Cluster Dashboard: web-based btop-inspired dashboard for multi-node Spark monitoring (paul.aviles) | https://forums.developer.nvidia.com/t/359975 | 2026-02-07 |

## Batch 5 forum sources (2026-07-10 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-mimo-sglang-4x | forum | MiMo-V2.5 (FP8) on 4-node DGX Spark via SGLang — recipe, parser bug, NCCL_CUMEM_ENABLE=0, EAGLE OOM, sampling params, tool eval 89/100 (mclenithan) | https://forums.developer.nvidia.com/t/368097 | 2026-04-27 |
| S-forum-diffusion-speeds | forum | Image diffusion speeds on GB10: Z-Image-Turbo/SDXL/Qwen-Image/ERNIE/FLUX.2-klein/Krea2 benchmarks, DIFFUSERS_ATTN_BACKEND=_native_cudnn, NVFP4 quant speedups (ijontichy) | https://forums.developer.nvidia.com/t/369095 | 2026-05-05 |
| S-forum-m27-recipe | forum | MiniMax-M2.7 NVFP4/AWQ/FP8 recipes & benchmarks on 2×/4× Spark — FlashInfer-CUTLASS vs CUTLASS, AWQ beats NVFP4 decode, FP8 4×=36 tok/s, AWQ 4×=41.6 tok/s (serapis, ekkis, aostang, miken, co-le) | https://forums.developer.nvidia.com/t/366324 | 2026-04-12 |

## Batch 6 forum sources (2026-07-11 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-nsight-remote | forum | Nsight Systems remote profiling on DGX Spark — SSH sudo requirement, no passwordless sudo for remote target (mt42) | https://forums.developer.nvidia.com/t/376266 | 2026-07-09 |
| S-forum-thermal-shutdown | forum | Sparks randomly powering off after long uptime — thermal paste degradation, CPU hot-spot sensor blind spot, repaste+case removal fixes; PDU fault variant (arctic.gus, Zatz, robin.s) | https://forums.developer.nvidia.com/t/376103 | 2026-07-08 |
| S-forum-3node-nccl | forum | 3-node DGX Spark NCCL failure — sparkrun auto detects switch vs ring, cable mixing (no-name 1500 MTU vs ASUS 9000 MTU), explicit SSH hostname resolution needed (nvidia4468, amurnane123, karol.spark) | https://forums.developer.nvidia.com/t/376215 | 2026-07-09 |

## Batch 7 forum sources (2026-07-11 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-mimo-2x-opt | forum | MiMo-V2.5 Flash on 2 Nodes optimization thread — detailed renek recipe (FA3 crash, TRITON_ATTN_DIFFKV guard bypass, driver KV pool diff, NCCL CGA buffer, MTP acceptance, 30-33 tok/s) + tonyd615 GitHub repo (38 tok/s, non-eager) | https://forums.developer.nvidia.com/t/373669 | 2026-06-18 |
| S-forum-onboarding | forum | DGX Spark first-boot onboarding: WiFi setup SSID never broadcasts on some units, QR code → product page not setup guide, monitor+keyboard workaround | https://forums.developer.nvidia.com/t/376293 | 2026-07-10 |

## Batch 8 forum sources (2026-07-12 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-clock-5min | forum | GPU clock wedge follow-up: 5 min power-off wait sufficient (was 30 min); power-drain method (hold power button 5-10s, no wait); root cause hypothesized in PSU power-control circuits stuck in safety protocol (florin.andrei, 0rand) | https://forums.developer.nvidia.com/t/376239 | 2026-07-09 |

## Batch 9 forum sources (2026-07-12 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-reboot-powercycle | forum | DGX Spark reboot doesn't complete — shuts down but doesn't power back on; requires USB-C cable removal + reinsert; intermittent (full shutdown + power button also works) (jp176) | https://forums.developer.nvidia.com/t/376431 | 2026-07-10 |
| S-forum-cx7-dual-setup | forum | Two DGX Sparks over CX-7 direct link field report — 200G QSFP56 DAC works, iperf3 TCP ~16 Gb/s (Grace CPU ceiling), SSH ~600 MB/s, DCGM works on GB10 (Xid + PCIe replay), PSI+swap-out OOM alerting, cluster tax metric, Amphenol NJAAKK-N911 certified cable ID (griffith.mark, mashie) | https://forums.developer.nvidia.com/t/376298 | 2026-07-10 |

## Batch 10 forum sources (2026-07-13 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-nvfp4-worth | forum | NVFP4 quantization on Spark — NVIDIA refreshed build.nvidia.com recipe; Qwen3/Qwen3.6 27B hit TensorRT-LLM errors; community gist recipe (paul448) | https://forums.developer.nvidia.com/t/376530 | 2026-07-11 |
| S-forum-nvidia-vfx | forum | No nvidia-vfx (Maxine VFX SDK) aarch64 wheel for DGX Spark — GB10 not in supported GPU list; NVIDIA confirmed no plans; ComfyUI RTX nodes broken (paulsc.liu, aniculescu) | https://forums.developer.nvidia.com/t/363267 | 2026-03-12 |

## Batch 11 forum sources (2026-07-13 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-easy-vllm | forum | easy-vllm code-agent harness for vLLM on DGX Spark — DSV4-Flash on GB10 via jasl/vllm SM12x fork (PR#41834, SHA c766cbc6), --moe-backend humming, NVML clock telemetry patch, MXFP4 MoE→MARLIN-repack→UMA OOM, torch 2.11+ ABI wall, ib_write_bw 208-218 Gb/s, mem_watchdog+earlyoom (sh.ahn) | https://forums.developer.nvidia.com/t/376574 | 2026-07-11 |
| S-forum-4node-crs504 | forum | 4-node DGX Spark cluster with CRS504 switch: DSV4-Flash TP=4 52-53.6 tok/s, M3-AWQ TP=4+EAGLE 28-35 tok/s, 100G vs 200G link (5-10% PP loss, no decode change), measured traffic ~13 Gb/s, $25 100G cable works (CosmicRaisins, corbett_korbett) | https://forums.developer.nvidia.com/t/373818 | 2026-07-11 |

## Batch 12 forum sources (2026-07-14 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-tokenspeed | forum | TokenSpeed SM12x-stable for DSV4-Flash on 2× Spark: prefill +10-14% vs vLLM fork, KV +25%, decode behind 70-74%; torch 2.13, FlashInfer CUTLASS MXFP4 MoE; NCCL 2.30.4 mandatory (jasl) | https://forums.developer.nvidia.com/t/369218 | 2026-07-12 |
| S-forum-spark-studio | forum | Spark Studio: open-source inference dashboard for DGX Spark — vLLM/SGLang/llama.cpp/sparkrun recipes, live UMA monitor, pre-launch memory guard, agent auto-fix, multi-node view, MIT (TheAwakenOne) | https://forums.developer.nvidia.com/t/376507 | 2026-07-10 |
| S-forum-dsv4-kvcache | forum | DeepSeek-V4-Flash KV cache ~15 GB/1M tokens/node on 2× Spark, vLLM CUDA graph memory profiling overhead, discrepancy with online KV calculators (paxren2020) | https://forums.developer.nvidia.com/t/376591 | 2026-07-11 |

## Batch 13 forum sources (2026-07-14 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-acestep-music | forum | ACE-Step v1.5 XL music generation on single Spark — fits comfortably in VRAM; 5Hz-LM-4B companion model for lyrics; 3 independent users confirm (danielgbates, joey28, aostang) | https://forums.developer.nvidia.com/t/376653 | 2026-07-12 |

## Batch 14 forum sources (2026-07-15 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-unsloth-qwen36 | forum | Unsloth Qwen3.6 NVFP4 quants on DGX Spark — 35B-A3B Unsloth NVFP4 ~15% slower than nvidia NVFP4 on GB10 (3 independent benchmarks); 27B fits 24GB; flashinfer_b12x unavailable on stock vLLM (falls back to Marlin); working spark-vllm-docker recipe with Marlin MoE + MTP; W4A16 bypass hypothesis (emX0r, hedelyuk.alexandr, J-R, TheAwakenOne, jbourny, azampatti, robert287) | https://forums.developer.nvidia.com/t/376484 | 2026-07-10 |
| S-forum-dsv4-vision | forum | DeepSeek V4 + vision model co-hosting on 2× Spark — memory balancing: DSV4 context cut to 256K @ 0.73 util to fit Qwen3-VL; recommended offloading vision to separate machine (MacBook, etc.); gpieceoffice runs Gemma-12B-NVFP4 + Qwen3.5-9B-FP8 as multimodal front-end + DSV4 MTP=3 as text reasoning, 35-40 tok/s combined (cerchez07, StarChickenXVII, 0rand, gpieceoffice) | https://forums.developer.nvidia.com/t/376790 | 2026-07-14 |
| S-forum-llama32-finetune | forum | Llama 3.2 3B full fine-tuning 8× slower than benchmark — 0.59 steps/s vs expected ~5 steps/s; NVIDIA redirect to DGX Spark Performance FAQ + benchmarking guide (arijitmukh007, raphael.amorim) | https://forums.developer.nvidia.com/t/353011 | 2025-11-30 |
| S-forum-hpc-slurm | forum | HPC/slurm/MPI on DGX Spark — NVIDIA Deepops all-in-one slurm on single Spark; CPU topology NUMA (Cortex-X925 perf + Cortex-A725 efficiency cores); P/E core binding for slurm partitions; CX-7 switch topology needs special config; enroot/pyxis containers for GenAI; RoCE not real IB (pavuknm, bugsareyummy, dbsci, paul448) | https://forums.developer.nvidia.com/t/366724 | 2026-04-15 |

## Batch 15 forum sources (2026-07-15 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-cx7-hotplug | forum | ASUS GX10 ConnectX-7 not showing in ifconfig/lspci — CX-7 ports are hot-pluggable; controlled by /etc/nvidia/cx7-hotplug-enabled; idle power draw nearly doubles when cable connected (mhoare1984, elsaco, mashie) | https://forums.developer.nvidia.com/t/376825 | 2026-07-14 |
| S-forum-llm-comfyui | forum | Running large LLM on 2× Spark cluster while keeping ComfyUI usable — vLLM KV cache reserve starves co-hosted workloads, --gpu-memory-utilization 0.7-0.8 workaround, llama.cpp better for co-hosting than vLLM, ComfyUI on head node works with reduced util (Alexander-F, AakankshaS, clawdiusmaximus, C_G, vasimv) | https://forums.developer.nvidia.com/t/376650 | 2026-07-12 |
| S-forum-qwen397-arch | forum | Architecture analysis: Qwen3.6-397B upcycling feasibility — interconnect bottleneck (~23 GB/s cross-node vs ~600 GB/s in-box), MoE all-to-all sensitivity, FP8 training impossible on sm_121, Megatron on GB10 caveats (vedcsolution, raphael.amorim) | https://forums.developer.nvidia.com/t/369561 | 2026-05-09 |

## Batch 16 forum sources (2026-07-16 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-colibri-glm52 | forum | Colibri engine: pure C, zero deps, streams GLM-5.2 (744B MoE) experts from disk on single DGX Spark — 2.4 tok/s full top-8, 3.33 tok/s with experimental CACHE_ROUTE; O_DIRECT disk I/O 9.69 GB/s; int4 MoE + int8 MTP heads; COLI_CUDA_UNIFIED=1 (Jcagle, Keving; benchmark by VincentMarquez via GitHub issue #161) | https://forums.developer.nvidia.com/t/376749 | 2026-07-13 |
| S-forum-comfyui-optimized | forum | ComfyUI Docker optimized for DGX Spark — CUDA 13.1 base, PyTorch cu130, SageAttention 2 compiled for sm_121, Comfy Kitchen NVFP4; double-VRAM bug fix (copy=False in tensor.to() with --disable-mmap); --disable-dynamic-vram; cudaMemGetInfo under-reports free UMA when co-resident CUDA process — fix via psutil.virtual_memory().available (luix93, Haidij) | https://forums.developer.nvidia.com/t/364846 | 2026-03-26 |

## Batch 17 forum sources (2026-07-16 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-nvfp4-broken | forum | NVFP4 on GB10 meta-analysis — 9× Spark customer: TRT-LLM NVFP4 slower than GGUF Q4_K_M, NVFP4 leaves ~half layers bf16, bandwidth efficiency 42-48%, flashinfer 0.6.8.1 merged, NVFP4 now operational via community Docker (DropTheBeat, tenari, jwarner, whpthomas) | https://forums.developer.nvidia.com/t/367082 | 2026-04-19 |
| S-forum-dsv4-abliterated | forum | DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored on 2× Spark TP=2 — fork of DS4 DSpark recipe with model swapped in, 50-60 tok/s, abliterated (uncensored) variant (tonyd615) | https://forums.developer.nvidia.com/t/376500 | 2026-07-10 |
| S-forum-nemotron-ollama | forum | Nemotron-3-Super 120B on Ollama v0.30.x-v0.31.2 parser regression — SSE stream aborts mid-response, no finish_reason; fix: downgrade to Ollama 0.24.0; v0.31.2-rc1 does NOT fix (frank.stockmans) | https://forums.developer.nvidia.com/t/375835 | 2026-07-07 |
| S-forum-ibwrite-false | forum | ib_write_bw falsely reports >64 KiB RDMA WRITE failure on GB10 — fabric is fine; minimal libibverbs probe passes all sizes; NCCL_NET_PLUGIN=none, NCCL_TOPO_FILE correction, RoCE NIC-offloaded counters, arp_ignore=1/arp_announce=2 (noc19) | https://forums.developer.nvidia.com/t/375603 | 2026-07-05 |

## Batch 18 forum sources (2026-07-17 ingest)

||| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
|| S-forum-glm52-8x | forum | GLM-5.2-Int4-Int8Mix (QuantTrio) on 8× GB10 TP8 DCP=1 — ~1,200 t/s prefill, 33–54 t/s avg decode; v16-unified branch (local-inference-lab/vllm 5dffea8), b12x W4A8 MoE (lukealonso/b12x 97b3d64), 4-patch set, DCP4 decode-starvation scheduler (penguinchang), NCCL_BUFFSIZE 16 MB, draft_tp=1, VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1; TP4+PP2 MTP collapses to ~8% (ciprianveg, penguinchang) | https://forums.developer.nvidia.com/t/376831 | 2026-07-14 |
|| S-forum-bonsai27b | forum | Qwen3.6-27B Binary/Ternary (Bonsai 27B) by Prism-ML — 1-bit and ternary builds, 94% quality claim, much smaller footprint; hypothesis: faster decode on bandwidth-bound Spark dense, esp. with MTP; no GB10 benchmarks yet (nerhun, m0l0, stu.miller, robert287) | https://forums.developer.nvidia.com/t/376879 | 2026-07-15 |

## Batch 19 forum sources (2026-07-17 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-usb2-fallback | forum | USB3 SuperSpeed PHY not registered — all USB falls back to 480 Mbps USB 2.0; MediaTek T-PHY (phy-mtk-tphy) has no ACPI binding; FE vs GX10 behavioral split (rstovall, elsaco, paulsc.liu, rob-engassist, al9999, pontostroy) | https://forums.developer.nvidia.com/t/362015 | 2026-03-01 |
| S-forum-fw-july2026 | forum | New FE Spark firmware: EC 0x03000302→0x03000508, UEFI SoC 0x0200980f→0x02009b0b; fwupdmgr not seeing update initially (elsaco, vasimv, mrDragonFox) | https://forums.developer.nvidia.com/t/376890 | 2026-07-15 |
| S-forum-ota-loop | forum | DGX Dashboard July 2026 software update stuck in loop — manual `apt upgrade` workaround; nvidia-spark-ota-check diagnostic tool; nv-docker-options missing (andybchen131, elsaco) | https://forums.developer.nvidia.com/t/376981 | 2026-07-15 |
| S-forum-asus-fw0103 | forum | ASUS Ascent GX10 BIOS/Firmware v0103 — PD/0x507 capsule update fixed 4× inter-Spark link speed, lower temps, ~8-10 W less; July 2026 OTA loop on Asus (brian322, trithemius, btvd, robert287, elsaco) | https://forums.developer.nvidia.com/t/364160 | 2026-03-20 |
| S-forum-host-freeze-tp2 | forum | Total host freeze (not process hang) during multi-node TP=2 vLLM prefill on 2× Spark — Step-3.7-Flash-NVFP4; zero forensic trace across kdump/watchdogs/netconsole; diagnosed as thermal shutdown (heathen0711, jrsphd) | https://forums.developer.nvidia.com/t/376882 | 2026-07-15 |

## Batch 20 forum sources (2026-07-18 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-mtp-lossless | forum | MTP lossless? — quality debate: MTP measurably affects output quality (tool-call bench ~5 pts, Qwen3.6-27B ~2% hit); vLLM+llama.cpp MTP+prefix-cache interaction bugs; DS4F prefix-batch 16384/MTP4 → 70-75% acceptance; "theory != deployment" practical-lossiness argument (JasonW, Nerhun, A3refaat, Azampatti, 0rand, mangosq) | https://forums.developer.nvidia.com/t/377030 | 2026-07-16 |
| S-forum-machineid | forum | MSI EdgeXpert DGX Sparks ship with identical /etc/machine-id (and identical SSH host keys) — CVE-2026-24218; ASUS GX10 same; OEM clone-image not sanitized; one-liner fix; MSI patched May 2026 (ohaibuzzle, emptysands, JW2026) | https://forums.developer.nvidia.com/t/377208 | 2026-07-17 |
| S-forum-nm-phantom | forum | NetworkManager "Connection failed" popup on DGX Spark — phantom DHCP profiles auto-created for ConnectX QSFP ports retry every ~45 s when carrier present but no DHCP server (Spark-to-Spark direct cable typical); fix: nmcli connection.autoconnect no on looping profiles; ip-config-unavailable = has link, no lease (YolandaHuang) | https://forums.developer.nvidia.com/t/377220 | 2026-07-17 |

## Batch 22 forum sources (2026-07-19 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-ec-fan-rollback | forum | EC firmware 0x0300xxxx breaks fan curve on DGX Spark → thermal throttling (96-97°C ACPI zones, inaudible fans); EC isolates fan control from OS (fancontrol/pwmconfig/nvidia-settings can't override); fix: fwupdmgr downgrade to 0x02004e18; idle 60→32°C, load 35-37°C, 0% throttling, 120-125W/node @ 95% GPU util; avoid blanket fwupdmgr update afterward (veelacleave, JW2026) | https://forums.developer.nvidia.com/t/377069 | 2026-07-16 |
| S-forum-nemo-rt | forum | Nemo-RT Community: real-time bilingual ES/EN voice agent (VAD+STT+LLM+TTS) co-located on one GPU, OpenAI Realtime API-compatible; on DGX Spark GB10 ~20 concurrent calls sub-second TTFA; Qwen3-8B-FP8 via vLLM; native FP8 + arm64 build; Apache-2.0 (InfinitoCloud) | https://forums.developer.nvidia.com/t/376248 | 2026-07-09 |
| S-forum-litellm-orchestrator | forum | harinezumigel-llm-stack: LiteLLM + NVIDIA vLLM Docker orchestrator for managing multiple local models on single DGX Spark — config.yaml + .env, container reuse, single OpenAI-compatible endpoint; multi-model lifecycle tool (HarinezumIgel); thread also surfaces Spark Studio + sparkstation (kshetrajna12/sparkstation) | https://forums.developer.nvidia.com/t/376407 | 2026-07-10 |

## Batch 23 forum sources (2026-07-19 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-sync-locale | forum | NVIDIA Sync / Cluster Assistant fails "Software version" check on non-English locale (German de_DE.utf8) — apt-cache policy parser looks for "Installed:" but localized output says "Installiert:" → false "System Software Update Required"; workaround: sudo update-locale LC_MESSAGES=en_US.utf8; hotfix pending (paul.oesterwitz, aniculescu/NVIDIA) | https://forums.developer.nvidia.com/t/377079 | 2026-07-16 |
| S-forum-ec-fan-asus | forum | ASUS GX10 thermal throttling after EC 0x02000005 / UEFI 0x03000006 update — corroborates S-forum-ec-fan-rollback on a 3rd OEM SKU; ACPI zones 96.6°C, GPU 85-90°C, SW/HW thermal slowdown counters, fans N/A; EC 0x02000004 vs 0x02000005 fan curve byte-identical (48%@85°C, 54%@93°C, 68%@95°C, 100%@97°C) → root cause may be SoC/UEFI interaction not EC table; no fwupdmgr downgrade available for ASUS GX10; fieldiag 2.0.4-1 packaging bug (ofed-scripts missing); NVIDIA escalated (giunta.francesco, veelacleave, Neill/NVIDIA) | https://forums.developer.nvidia.com/t/377044 | 2026-07-16 |
| S-forum-inkling | forum | Inkling 975B (41B active) MoE + Inkling-Small 276B (12B active) multimodal model announcement — 1M context, text/image/audio/video; community plans 8× Spark cluster bring-up; no recipe/benchmarks yet (eh17, greg190) | https://forums.developer.nvidia.com/t/377238 | 2026-07-17 |

## Batch 24 forum sources (2026-07-20 ingest)

||| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-6x-cluster | forum | 6× GB10 cluster via MikroTik CRS812 (768 GB combined) — b12x backend enables TP=6 on most models; GLM-5.2 ~30 tok/s single-stream; cluster peak 800-1180 W (mclenithan) | https://forums.developer.nvidia.com/t/376585 | 2026-07-11 |

## Batch 25 forum sources (2026-07-20 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-inkling-nvfp4 | forum | Inkling (Thinking Machines) NVFP4 on 8× DGX Spark — full bring-up: NVFP4 clean (no dtype fallbacks), cudagraphs working (boundary bug in Sm120 rel-bias attention kernel root-caused via GPU coredump), paged-KV absent in tml_fa4 Sm120 path (workaround re-gathers KV per decode step), LAMPORT_RS_SCONV=0 escape hatch for RoCE clusters (Lamport collectives require MNNVL/NVLink), vllm#49049 filed; decode 25-27 tok/s short ctx, drops to 13.5 tok/s @ 2048 ctx (c1); prefill 1400-2711 tok/s; MTP k=1 stuck (60% draft acceptance); recipe + 12 patches at blockmos/inkling-sparks-gb10; parked in favor of M3 (greg190, vexus777) | https://forums.developer.nvidia.com/t/377306 | 2026-07-17 |
| S-forum-kimi-k3-ceiling | forum | Kimi K3 (2.8T) & the practical GB10 cluster ceiling — cluster sizing math: ~115 GB usable/node → 16 nodes for K3 @ 4-bit (~$100k, 2000-3200W); ~50B active → 35-50 tok/s projected; viable 200B-class alternatives list (Step-3.7-Flash, Command-A-Plus, Inkling-Small, Laguna-M.1, Qwen3.5-397B-A17B, Hy3); mashie's switch-less 5-node full-mesh via MST sub-port splitting (break 4x50G → 2x50G per QSFP port, 6 RoCE interfaces for 5 nodes), ~$800 optical transceiver cost vs MikroTik (CosmicRaisins, jwarner, mashie, danielgbates) | https://forums.developer.nvidia.com/t/377091 | 2026-07-16 |
| S-forum-intern-s2 | forum | internlm/Intern-S2-Preview-397B — 397B model announced as preview (Claude Opus-4.8/GPT-5.5 class benchmarks claimed); no quantization small enough for 2× Spark exists yet; community requests a 4× Spark / autoround recipe; no GB10-specific config or benchmarks (chrm, Sparkdown_Format) | https://forums.developer.nvidia.com/t/377342 | 2026-07-18 |
| S-forum-pmu-amu | forum | GB10 Spark ARM PMU/AMU counters — Cortex-A725 capped to 1 GHz, Cortex-X925 up to ~1375 MHz; correct PMU event for Spark differs from ARMv8; kernel module for reading PMC counters now available (CyrIng project master branch); A725 factory 2.8 GHz vs X925 scaling anomaly when normalizing PMC reads to max core freq (CyrIng) | https://forums.developer.nvidia.com/t/377280 | 2026-07-17 |

## Batch 26 forum sources (2026-07-21 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-vllm025-nccl | forum | vLLM 0.25.1 + NCCL 2.30.7 not available in standard Spark Docker images — community images ship vLLM 0.23 + old NCCL; latest needed for PP (MiniMax M3) and mesh NCCL; GitHub image with latest found (Hunlx) | https://forums.developer.nvidia.com/t/377417 | 2026-07-19 |
| S-forum-flashinfer-livelock | forum | FlashInfer sparse_mla_sm120 kernels livelock on GB10/sm_121 under cold-prefill — mbarrier TRYWAIT phase check spin-loop root-caused via cuda-gdb; validated workaround: Triton sparse-MLA (FLASHMLA_SPARSE + sm12x Triton patch); GLM-5.2 TP=4 4× Spark; 560+ clean sessions post-workaround; evidence pack at marksunner/glm52-dgx-spark-deadlock-evidence (msunner) | https://forums.developer.nvidia.com/t/377334 | 2026-07-18 |
| S-forum-3node-mesh | forum | 3-node Spark mesh networking guide (spark-vllm-docker + sparkrun) — CX-7 full mesh without switch, cross-connect port0↔port1; TP requires power-of-2 (attention head divisibility); 3-node PP slower than 2-node TP=2; LMCache for dedicated KV cache node; NCCL mesh merged to main; Qwen3.5-397B-A17B-int4-AutoRound benchmarks; vLLM PP+MTP not supported; fastsafetensors freeze fix; gpu_memory_utilization 0.8 stable (eugr, dbsci, chunkai721, jameslacroix) | https://forums.developer.nvidia.com/t/365296 | 2026-04-01 |
| S-forum-update-loop | forum | DGX Dashboard update loop — EC firmware 0x00000500→0x00000507 fails silently; fwupdmgr get-results shows Update State: Failed; power-cycle workaround (unplug USB-C brick, hold power 10s, wait 5 min); 2-3 cycles may be needed (podstawek, jcagle) | https://forums.developer.nvidia.com/t/363464 | 2026-03-14 |

## Batch 27 forum sources (2026-07-21 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-temps-normal | forum | DGX Spark thermal zones under load: 7 sysfs acpitz zones, zones 0/5 hit 94.6°C, GPU ~10°C cooler than CPU; tegrastats (Jetson Orin Nano binary) works on Spark; wildpines.ai clock-capping blog referenced (DannyTup, sjug, elsaco, digirho) | https://forums.developer.nvidia.com/t/377375 | 2026-07-18 |

## Batch 28 forum sources (2026-07-22 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-uvm-livelock | forum | Spark abrupt shutdown under sustained load (Qwen 3.5 122B, 35B+27B co-loaded) — UVM page-migration livelock: weights + KV cache + CUDA workspace share 128 GB pool, over-commitment causes hard-lock with no OOM-killer, no log; fix: --gpu-memory-utilization 0.85–0.92, don't co-load large models, leave ~10–15 GB free; platform firmware (BIOS/BMC) update; nvidia-smi -pm 1 + -pl power cap; nvidia-smi -lgc clock lock; PSU overheating on carpet (stuart.trusty, mbnshahrzad, oddjobsandservices, aniculescu) | https://forums.developer.nvidia.com/t/377478 | 2026-07-20 |
| S-forum-sway-scanout | forum | NV_ERR_NO_MEMORY in Sway compositor on GB10 — memmgrAllocScanoutCarveoutRegionResources_GB10B fails allocating scanout carveout from UMA pool; 6144×3456@60Hz = ~121 MB/buffer, multiple buffers need several hundred MB contiguous; UMA fragmentation at boot causes failure with <4 GB/122 GB used; WLR_SCENE_DISABLE_DIRECT_SCANOUT=1 doesn't fix; driver 580.142, CUDA 13.0 (dlludllu, parallelArchitect) | https://forums.developer.nvidia.com/t/370458 | 2026-05-18 |
| S-forum-sparkdash-mia | forum | sparkDash by MiaAI-Lab — open-source multi-DGX Spark monitoring dashboard: live GPU/CPU/unified memory/storage/network, local LLM status (llama.cpp, vLLM, sglang) with tok/s, SSH power controls + Wake-on-LAN, worker-node flag; trusted LAN (no built-in auth) (MiaAI_Lab) | https://forums.developer.nvidia.com/t/377550 | 2026-07-20 |
| S-forum-realsense-d435 | forum | RealSense D435 USB disconnect on Dell GB10 (DGX Spark) — kernel disconnects, unplug/replug doesn't fix, rmmod/modprobe doesn't fix, only reboot; librealsense2 v2.56.5/v2.57.4 RSUSB; fixed by July 2026 firmware update (qobi, aniculescu) | https://forums.developer.nvidia.com/t/351088 | 2025-11-11 |

## Batch 29 forum sources (2026-07-22 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-6x-ring-rdma | forum | 6-node DGX Spark ring topology NCCL — RoCE L2-adjacency requirement, NCCL_IB_MERGE_NICS=0 + NCCL_IB_SUBNET_AWARE_ROUTING=1 fix, nvidia-peermem refuses to load (no GPUDirect), Qwen3.6-35B-A3B-NVFP4 PP=6 ~21 tok/s/request, RDMA vs TCP only ~7% gain (alpaslan.erdag, Hunlx, mashie) | https://forums.developer.nvidia.com/t/377435 | 2026-07-19 |
| S-forum-uefi-fw-fail | forum | UEFI firmware update failing — fwupdmgr reports bad PD firmware version 0x00000001, installed version can't bridge to current; stepping-stone firmware needed; dmidecode -t 45 diagnostic (dmaynor, aniculescu/NVIDIA, lewdenlw) | https://forums.developer.nvidia.com/t/365116 | 2026-03-30 |
| S-forum-serial-console | forum | DGX Spark serial console — NVIDIA confirms not supported, removed from Porting Guide (ragge, aniculescu/NVIDIA) | https://forums.developer.nvidia.com/t/369350 | 2026-05-07 |
| S-forum-sleep-disabled | forum | Sleep/suspend disabled by default on DGX OS — NVIDIA staff confirms, overrideable (allanmac, aniculescu/NVIDIA) | https://forums.developer.nvidia.com/t/377582 | 2026-07-20 |

## Batch 30 forum sources (2026-07-23 ingest)

||| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-mistral-s4-119b | forum | Mistral Small 4 119B NVFP4 on DGX Spark (GB10) — 67-post thread; MLA head_size=320 rejected by all backends initially, TRITON_MLA resolves it; working recipe TRITON_MLA + NVFP4 + FLASHINFER_CUTLASS MoE + fp8 KV; 28-33 tok/s decode (5 independent reporters); reasoning_effort bug, tool-calling PR #39217, Eagle/MTP not working, --shm-size 16g kernel crash, vLLM 0.25.1 native arm64 images (chuckchambersdev, mrDragonFox, cosinus, tenari, 0rand, drew22) | https://forums.developer.nvidia.com/t/363863 | 2026-03-17 |
| S-forum-qwen36-fp8-2x | forum | Qwen3.6-35B-A3B-FP8 on 2× Spark TP=2 — 75-80 tok/s output via spark-vllm-docker run-recipe.sh; FlashInfer, FP8 KV, 262K ctx, prefix caching, no-ray; cold TTFT 0.68s (5K ctx) / 8.49s (81K ctx); prefix cache kicks in hard on 2nd runs (gary100) | https://forums.developer.nvidia.com/t/373995 | 2026-06-21 |
| S-forum-cx7-dac-power | forum | CX7 DAC power/thermal — 6°C higher temps with DAC plugged in even after mlx5_core unbind + PCI remove; only physical DAC removal brings temps down; dgx-spark-mlnx-hotplug package manages CX7 via udev rules + MTKP0001 ACPI hotplug driver (meanaverage, raphael.amorim) | https://forums.developer.nvidia.com/t/366858 | 2026-04-17 |

## Batch 31 forum sources (2026-07-23 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
|| S-forum-spark-vllm-rebuild | forum | spark-vllm-docker build flags: --rebuild-vllm forces local image rebuild (vs pulling pre-built); --use-wheels uses prebuilt wheels instead of compiling vLLM from source; repo always builds from main (no pinned vLLM version) (elvis.dowson, eugr) | https://forums.developer.nvidia.com/t/376722 | 2026-07-13 |

## Batch 32 forum sources (2026-07-24 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-m3-tp3 | forum | MiniMax-M3 NVFP4 TP=3 on 3× DGX Spark — Luke Alonso chthonic vLLM+b12x virtual sharding (fb63c9a), 3 head-node OOM fixes (safetensors load, Ray object-store cap, Ray memory monitor disable), NCCL 2.30u1 baked LD_PRELOAD shim trap, cold power-drain fixes stuck ib_write_bw 12.8→111.85 Gb/s, 200K ctx over RoCE, ~6 tok/s over 1GbE (tonyd615, mashie, eugr_nv) | https://forums.developer.nvidia.com/t/373387 | 2026-07-23 |
|| S-forum-vllm-containers | forum | vLLM containers thread — NGC lags ~2 versions behind community; spark-vllm-docker nightly pre-built wheel pipeline with regression testing; --vllm-ref builds from source even without --rebuild-vllm; multi-container via --name (eugr, joshua.dale.warner, WillLee) | https://forums.developer.nvidia.com/t/362721 | 2026-07-22 |
|| S-forum-laguna-quality | forum | Laguna-S-2.1 on single Spark ~20-30 tps reasoning — quality as good as DSV4-Flash (2× Spark) for document reasoning+tools; fails single-shot HTML/simulation generation (alperen.duru17) | https://forums.developer.nvidia.com/t/377674 | 2026-07-22 |
|| S-forum-solar-open2 | forum | Solar-Open2-250B (250B-A15B MoE) INT4 on 2× Spark — ~15 tok/s decode (tg32), ~2227 tok/s prefill (pp2048), flat across depths to 32K; no MTP tested (FoRWiS) | https://forums.developer.nvidia.com/t/377765 | 2026-07-22 |

## Batch 82 forum sources (2026-08-21 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-cx7-qsfp-breakout | forum | Mellanox QSFP breakout on DGX Spark CX-7 — NUM_OF_PF and MODULE_SPLIT_M<x> params available but port splitting fails; only first QSFP lane establishes link with breakout cable; devlink reports splittable=false; only 200G QSFP112 DAC cables validated for Spark; mashie observes phantom links on unused split lanes (greg.pring, elsaco, aniculescu, isdias, steel-rat, mashie) | https://forums.developer.nvidia.com/t/349775 | 2025-11-01 |
| S-forum-75w-crash | forum | DGX Spark hardware crashes at ~75W, system reboots repeatedly — fix: nvidia-smi -lgc 300,2100 clock cap; release with nvidia-smi -rgc; corroborates existing [reported] clock-cap mitigation (cory.farr) | https://forums.developer.nvidia.com/t/380669 | 2026-08-20 |
| S-forum-fieldiag-signedby | forum | Field Diagnostics install fails with Signed-By conflict — documentation error in Field Diagnostics guide incorrectly instructs adding CUDA apt repo, but DGX Spark OS already ships with it under different keyring name (cuda_debian_prod.gpg vs cuda-archive-keyring.gpg); fix: sudo rm /etc/apt/sources.list.d/cuda-sbsa-ubuntu2404.list; NVIDIA staff (Neill) confirmed doc error; thread also reports idle overheating + system freezes (asdf.think365, Neill) | https://forums.developer.nvidia.com/t/380584 | 2026-08-19 |

## Adding a source

Append a row with a new `S-` id and its `type`, then ingest per [`../SCHEMA.md`](../SCHEMA.md)
and [`../agents/ingest.md`](../agents/ingest.md). Forum/repo/report sources cite a URL; first-party
sources cite the experiment (what/config/when), never a private path.

## Batch 33 forum sources (2026-07-24 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-qwen3tts-ggml | forum | Qwen3-TTS on DGX Spark: GGML CUDA crash `ggml_cuda_kernel_can_use_pdl` (PDL capability check) — root cause is CUDA 12.8/sm_120 build, not sm_121a; fix: force torch backend (CUDA graphs, no GGML); TTFA 2.65s, steady-state RTF ~1.7; PDL oddness on GB10 corroborated (swann.schilling, Drew_the_AI_Guy) | https://forums.developer.nvidia.com/t/377743 | 2026-07-22 |

## Batch 34 forum sources (2026-07-25 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-vllm-stock-hang | forum | Stock vllm/vllm-openai:latest hangs silently during model load on ASUS GX10 (GB10) — never reaches "Application startup complete"; root cause is no SM121/Blackwell support in stock image; fix: use spark-vllm-docker --tf5 or CUDA 13/SM121 wheel (dotrantrung2003, Drew_the_AI_Guy) | https://forums.developer.nvidia.com/t/377613 | 2026-07-24 |
| S-forum-locateanything | forum | LocateAnything-3B (visual grounding) bring-up on DGX Spark / ThinkStation PGX via spark-vllm-docker --tf5 — ARM64 wheel gaps (decord, deepspeed, bitsandbytes, liger_kernel), device_map='auto' slow on 128GB UMA, MoonViT sub-model HF auth hang, FastAPI server pattern for non-vLLM models (swann.schilling) | https://forums.developer.nvidia.com/t/371829 | 2026-07-23 |

## First-party sources (2026-07-22)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-laguna-v251-bench | first-party | Laguna-S-2.1-NVFP4 single-node TP=1 on vLLM 0.25.1 + FlashInfer nightly 0.6.15.dev20260712, DFlash spec=7; llama-benchy pp2048/tg128 depth sweep (d0/4096/8192/16384, 3 runs); decode 20-23 tok/s, prefill 3.2-3.9K tok/s, cold start ~15 min | first-party: single-node DGX Spark, 2026-07-22 | 2026-07-22 |
| S-forum-laguna-dflash | forum | Laguna-S-2.1-NVFP4 + DFlash on DGX Spark: decode 20-36 tok/s single-node, 40-50 tok/s with spec=7; DFlash acceptance 18-40% (7 tokens, structured/agentic), 2-3% (15 tokens); max_num_seqs=4 (default crashes); vLLM 0.25.0+ required (vr8vr8) | https://forums.developer.nvidia.com/t/377663 | 2026-07-19 |

## Batch 35 forum sources (2026-07-25 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-gridbook | forum | PrismaQuant GridBook (tenari/RobTand): vLLM plugin exposing a ladder of 41 codebook quant formats (1.781-6 bit, quarter-bit increments); dictionary entries constrained to FP8/NVFP4 grid for native tensor-core dequant via table lookup (~10% decode / 30% prefill overhead); releases Qwen3.6-27B 5.5-bit (KL 0.0049, 77% lower than AURA at same rate) and Hy3-295B-A21B 2.9-bit; new MTP-head quant optimizer; GGUF IQ formats on vLLM found lacking (platform-agnostic, poor prefill) (tenari, m0l0, chargeuk) | https://forums.developer.nvidia.com/t/377773 | 2026-07-22 |
| S-forum-nfs-modelshare | forum | NFS-share HuggingFace cache across cluster nodes via /etc/exports on head node + fstab automount on workers (ConnectX-7 IPs, not WiFi/Ethernet); docker save \| ssh load for image distribution; measured load speed ~7 Gbit/s (peaks 20 Gbit/s RAM-cached) over CX-7 vs slow DRAM-less 2242 NVMe; sparkrun has native NFS cache support + container drift detection; 4TB NVMe cooling concerns noted (Hunlx, dbsci, FlossingEnthusiast) | https://forums.developer.nvidia.com/t/377551 | 2026-07-20 |
| S-forum-mikrotik-cr804-042 | forum | MikroTik CRS804-4DDQ with 2× Spark via FS QDD-400G-2QPC02 breakout — link negotiates 200G, ping/TCP work, but NCCL/ib_write_bw stuck at ~0.5 GB/s vs expected 16-23 GB/s; packet_seq_err + rp_cnp_handled climb (DCQCN throttling) despite zero fabric drops; fix: full AC power-drain (unplug ~60s) for CX-7 firmware settings to apply; auto-negotiate may need explicit bandwidth setting for ~20-24 GB/s on 4× clusters (Thom.S, mashie, elsaco, joe.24x7) | https://forums.developer.nvidia.com/t/378042 | 2026-07-24 |
| S-forum-ling3-flash | forum | Ant Ling-3.0-Flash 124B-A5B announced (Ant Group/Alibaba): hybrid-linear attention (KDA:MLA 5:1 stack), 1/64 expert activation, 256K native ctx (scales to 1M); benchmarks beat their prior 1T model; weights "soon" after Aug 3; INT4 AutoRound/NVFP4 expected to contend with Qwen3.5-122B as single-Spark GOAT (entrpi, m0l0, xkm121) | https://forums.developer.nvidia.com/t/377903 | 2026-07-23 |
| S-forum-woolyai | forum | WoolyAI Private Multi-agent Inference Stack for 2× DGX Spark — multi-model agentic workflow server with scheduler that swaps resident models at safe boundaries; LlamaBenchy benchmarks unquantized, no spec decode: DeepSeek-V4-Flash C1 21.15 / C4 55.99 tok/s (14 per-req), Gemma-4-26B-A4B C1 30.22 / C4 63.75, Nemotron-3-Nano-Omni-30B-NVFP4 C1 39.42 / C4 90.83; model-activation-wait 2-16s; community skepticism (mrDragonFox): at C1 slower than llama.cpp, no launch command/repro recipe shared (manisha5, Drew_the_AI_Guy) | https://forums.developer.nvidia.com/t/377787 | 2026-07-22 |

## Batch 36 forum sources (2026-07-26 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-glm52-vision | forum | GLM-5.2-Vision-NVFP4 (baseten) on 4× DGX Spark — 49.5M-param projector maps MoonViT 1152-dim → GLM 6144-dim token space; text backbone + vision tower frozen byte-identical; adaptive MTP dynamically switches 2–5 drafted tokens based on p2–p4 acceptance; ported to CosmicRaisins/glm-5.2-gb10 repo; only OCR/classification tested (CosmicRaisins, ciprianveg) | https://forums.developer.nvidia.com/t/378101 | 2026-07-24 |
| S-forum-solar-open2-nvfp4 | forum | Solar Open2 250B (250B-A15B MoE, 36/48 KDA linear-attn layers) NVFP4 W4A4 on 2× Spark TP=2 — ~15.8 tok/s decode c1, flat across depths (15.4 @ 32k, −2.5%); FP8 KV speed-neutral but doubles pool (2.67M tok, 10.17× concurrency @ 262k); vLLM v0.25.1 from source sm121, UpstageAI fork v0.22.0-solar-open2; VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass, VLLM_USE_FLASHINFER_MOE_FP4=1 (danielgbates) | https://forums.developer.nvidia.com/t/378106 | 2026-07-24 |
| S-forum-typec-thermal | forum | DGX Spark overheating without load after July 23 firmware update — pending USB-C PD (type-C power) firmware update had not installed; 30-min full power-cycle cleared heat and installed the pending update; nvidia-smi -lgc 0,2000 suggested as clock cap workaround; NVIDIA staff requesting version numbers (unicornxoxo2, paulsc.liu, Neill/NVIDIA) | https://forums.developer.nvidia.com/t/378028 | 2026-07-24 |

## Batch 37 forum sources (2026-07-27 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-qwen122-king | forum | Best Daily Single Spark Driver — Qwen 3.5 122B "king model" consensus: 4 users confirm as single-Spark daily driver; AutoRound int4 ~65 tok/s 2× Spark (Josephbreda), ~35 tok/s fp8 1× (0rand), NVFP4 variants and autoround loop tendency discussed; sparkrun-recipes repo (styles01) — vLLM v26 patched, 5 lanes @ 256K ctx, 40+ tok/s decode; NVIDIA repo autoround loops; DS4-Flash 45-50 tok/s 1×, 240 tok/s @ 16 reqs; Mistral 119B used as daily driver (gaburko) (Styles01, Josephbreda, 0rand, Rerollingingenshitimpactsucks) | https://forums.developer.nvidia.com/t/378066 | 2026-07-24 |
| S-forum-sparkctl | forum | sparkctl: CLI and orchestration for managing model serving on DGX Spark single nodes and clusters — config-driven (YAML), multi-provider (vllm, ollama, llama.cpp), load balancing for clusters, data plane contextual (metrics, litellm proxy host/node/k8s); devops/k8s-inspired reproducible deployments; tutorial at bradmurry.com; community notes overlap with sparkrun (bradodarb, mrDragonFox) | https://forums.developer.nvidia.com/t/376858 | 2026-07-14 |

## Batch 38 forum sources (2026-07-27 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-glm52-hybrid | forum | GLM-5.2 Hybrid FP8+NVFP4+MXFP4 on 4× Spark — aidendle94 hybrid-quant checkpoint (~800 t/s prefill @100k, 25 t/s decode prose, ~800K ctx); custom NVFP4 KV cache w/ scaling+calibration; adaptive speculative depth; Docker image (sparkrun-vllm-ds4-gb10:production-hybrid-1.2/1.3); VLLM_PREFIX_CACHE_RETENTION_INTERVAL unset entrypoint workaround; llama-benchy depth/concurrency tables (alexander.korolev.germany); tool-eval-bench 86/100 v2, 85/100 v3-GPTQ; reasoning-parser structured-output 58% root cause (mike_ber: thinking-off → 100% SO, +8 pts); MTP5→MTP4 (83→85, ciprianveg); word-salad at >90k ctx traced to hardcoded repetition_penalty=1.2 (mclenithan); b12x sparse-MLA kernel only reads packed fp8 KV pages, bf16 KV → immediate EOS; FLASHMLA_SPARSE has no sm12x sparse kernels; NVIDIA official GLM5 NVFP4 ~115 GB/node weights, ~460 GB total (excl. 20 GB MTP) on 4× Spark (kevin.wu07); GPTQ-on-MXFP4 v3 model (aidendle94/GLM-5.2-MXFP4-Experts-GPTQ) (aidendle94, CosmicRaisins, alexander.korolev.germany, mclenithan, mike_ber, ciprianveg, kevin.wu07) | https://forums.developer.nvidia.com/t/377598 | 2026-07-21 |
| S-forum-asus-fw-jul25 | forum | New ASUS GX10 firmware (SoC + TPM updates) — stability solid at 96% load/70W+; 2-4% benchmark delta (within noise/fresh-state); slow reboot; concurrent minor nvidia driver update (robert287, J-R, AoE) | https://forums.developer.nvidia.com/t/378099 | 2026-07-24 |

## Batch 39 forum sources (2026-07-28 ingest)

||| ID | type | What it is | Reference | Date |
|---|---|---|---|---|---|
| S-forum-whisper-docker | forum | whisper.cpp STT server on DGX Spark via Docker — ARM64+CUDA build from source, CMAKE_CUDA_ARCHITECTURES="120;121" (not just 120), nvidia/cuda not nvcr.io (no ARM64 tags), CUDA stubs for linking, deploy.resources GPU access; whisperx-blackwell alternative (mekopa/whisperx-blackwell) (swann.schilling, ajvazan) | https://forums.developer.nvidia.com/t/371803 | 2026-05-30 |
| S-forum-qwen122-v26-dflash | forum | Qwen 122B vLLM v26 + fp8 KV + DFlash + int8 lm-head on single Spark — first working fp8 KV + DFlash on GB10 for hybrid quant models; 3 custom patches (inc_hybrid, int8_lmhead_v3, prefix_align); KV 549K→1.37M tokens (2.6×), concurrency 2.09×→5.24× @ 256K; decode 45.98 tok/s, prefill 957 tok/s (+32%); build vLLM from main (commit 318b527) (styles01) | https://forums.developer.nvidia.com/t/378167 | 2026-07-26 |
| S-forum-speedycolibri | forum | SpeedyColibri — Rust port of Colibri for GLM-5.2 (744B MoE) on single Spark; ~1 tok/s initial → ~4 tok/s with fp8; proof-of-concept by new developer; working on multi-spark (GriffinPilz/SpeedyColibri) (GPilz) | https://forums.developer.nvidia.com/t/376996 | 2026-07-16 |
| S-forum-llamacpp-fastest | forum | Fastest llama.cpp Docker image on DGX Spark — official ghcr.io/ggml-org/llama.cpp:full-cuda13 matches custom builds (72.28 tok/s tg128); --mmap 0 mandatory on UMA; Agents-A1-NVFP4-MTP-GGUF 35B-A3B NVFP4; performance degradation 40→67 tok/s fixed by system update (power-cycle pattern) (knitvoger1, pontostroy) | https://forums.developer.nvidia.com/t/376946 | 2026-07-15 |

## Batch 40 forum sources (2026-07-29 ingest)

||| ID | type | What it is | Reference | Date ||
---|---|---|---|---|---|
| S-forum-gemma4-26b-bench | forum | Gemma-4-26B-A4B NVFP4 benchmark: unsloth vs nvidia on DGX Spark vs RTX Blackwell 6000 Pro — vLLM serve, fp8 KV, 65K ctx, 100 concurrent reqs; Unsloth ~17% faster than nvidia on Spark (160 vs 128 tok/s aggregate output); Spark ~6-7× slower than Blackwell 6000 Pro; single-stream TPOT ~47-59 ms (shahizat) | https://forums.developer.nvidia.com/t/377364 | 2026-07-18 |
| S-forum-comfyui-crash | forum | ComfyUI hard-crash fix on DGX Spark — GPU power spike (14→85W) trips overcurrent protection, not thermal/OOM; fix: swapoff -a + nvidia-smi -lgc 300,2100 (clock cap to 2100 MHz, ~50W); --highvram is a trap on UMA (forces all models pinned); CUDA_CACHE_MAXSIZE=4GB gives 3× rerun speedup; async weight offload near-free on UMA; 2nd user confirms clock cap stabilizes (jas.burton, frozenace88) | https://forums.developer.nvidia.com/t/360336 | 2026-02-11 |
| S-forum-cuda-mps | forum | CUDA MPS for multiple vLLM instances on single DGX Spark — nvidia-smi -c EXCLUSIVE_PROCESS, nvidia-cuda-mps-control daemon, --gpu-memory-utilization 0.45 per instance; latency increases significantly, throughput improves modestly; enables serving multiple independent models on same GPU (shahizat) | https://forums.developer.nvidia.com/t/376724 | 2026-07-13 |

## Batch 41 forum sources (2026-07-29 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-power-90w | forum | Hard power-off under sustained GPU load at ~90W — detailed reproduction with stepped FP16 matmul, throttle bit logging; persists after full platform firmware update (SOCFW/EC/USBPD); clock cap 2200 MHz fixes; CPU 92-97°C while GPU 78-83°C; no orderly shutdown, no pstore, no rasdaemon errors; NVIDIA confirms known issue (pacardenaz, aniculescu) | https://forums.developer.nvidia.com/t/378315 | 2026-07-27 |
| S-forum-gpu-throttle-cmd | forum | GPU clock cap commands reference — nvidia-smi -lgc 0,2000; full speed ~80W → 2000MHz ~60W; performance basically unaffected at 2150MHz throttle (elsaco, azampatti) | https://forums.developer.nvidia.com/t/378300 | 2026-07-27 |
| S-forum-unsloth-b12x | forum | Unsloth vs nvidia Qwen3.6-35B-A3B-NVFP4 benchmark with flashinfer_b12x on Spark — Unsloth+b12x ~8% faster than nvidia+Marlin (436 vs 404 tok/s agg @100 conc); flashinfer_b12x working recipe (CUTE_DSL_ARCH=sm_121a, vllm>=0.25.0, flashinfer>=0.6.13); vLLM 0.25.x startup hang reported (shahizat, TheAwakenOne, rtamax) | https://forums.developer.nvidia.com/t/376703 | 2026-07-13 |
| S-forum-nvfp4-kv | forum | NVFP4 vs FP8 KV cache on DGX Spark and RTX 6000 Pro — SGLang, Qwen3-4B; NVFP4 KV gives 1.68× more capacity than FP8 on Spark (2.31M vs 1.37M tokens); dtype torch.float4_e2m1fn_x2; production should validate quality before enabling (shahizat) | https://forums.developer.nvidia.com/t/377425 | 2026-07-19 |
| S-forum-dsv4-reap25 | forum | DeepSeek-V4-Flash REAP25 PrismaAURA measured-quant for single GB10 — ds4-server fork, 92/100 tool-eval, 16.5 tok/s spec decode, 77.2% DSpark acceptance; IQ2_XXS+MXFP4+MXFP8 mixed quant via measured-KL knapsack; W4A8 CUTLASS type-40 source-faithful path; IQ2 stays on dp4a (tensor core dequant net loss); marco.palaferri fork: 854 tok/s prefill, 24-25 tok/s decode at 55k-70k ctx (twaggs88, marco.palaferri) | https://forums.developer.nvidia.com/t/376872 | 2026-07-14 |

## Batch 42 forum sources (2026-07-30 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-driver580-173 | forum | apt upgrade to driver 580.173.02 breaks GPU on OTA2607 — Xid 119 GSP_INIT_DONE timeout, SEC2 secure-boot fails; nvidia-spark-ota-check reports "torn" driver/firmware pairing; 2 units fail identically; 1 user reports 580.173.02 works on 4 Sparks (firmware-dependent); DGX Dashboard re-update fixes (chenette, amurnane123, padrian, aniculescu) | https://forums.developer.nvidia.com/t/378200 | 2026-07-26 |
| S-forum-model-storage | forum | Asus GX10 model storage strategies — USB3 falls back to USB2 if drive connected at boot (corroborates S-forum-usb2-fallback); USB SSD speed drops to 20 MB/s intermittently; NFS over 10GbE ~1.1 GB/s; 4TB NVMe upgrade (Corsair MP700); NVMe-oF over 400G fabric; modelctl tool; cron-based model offloading (starkrun, FlossingEnthusiast, ajvazan, danielgbates, robert287, nightonthesun, piresbruno) | https://forums.developer.nvidia.com/t/378310 | 2026-07-27 |
| S-forum-acer-thermal | forum | Acer Veriton GN100 thermal A/B test — 2 units running Qwen3.5-122B-A10B INT4 AutoRound + DFlash, 1h sustained load; both ~68°C under load, 96% GPU util, ~25 tok/s, zero throttling; idle gap didn't persist under load; Acer peaks ~68°C vs 80-82°C other OEMs; spark_hwmon driver referenced (jjustice, azampatti) | https://forums.developer.nvidia.com/t/378210 | 2026-07-26 |

## Batch 43 forum sources (2026-07-30 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-sm121-support | forum | DGX Spark (SM121) software support is severely lacking — 43-post thread; NVIDIA official response (johnny_nv) with version roadmap + community fact-checking (baristankut); vLLM --enforce-eager 20-30% perf loss, CuTE DSL FP4 restricted to sm_100a (Issue #2800), SGLang unofficial branch (sglang#11658), PyTorch 2.10 + FBGEMM/CUTLASS for sm12x, Triton 3.6.0 RC, FlashInfer 0.5.3+/0.6.1, CUTLASS 4.2.0+/4.3.5/4.4.x, vLLM 0.14.0 expected, MoE kernels no optimized GB10 configs, tcgen05/DSMEM/TMEM/TMA/multicast lacking, CUDA 12.0f vs 12.1a distinction (baristankut, johnny_nv, christopher_owen, vegax87, trystan1, josephbreda) | https://forums.developer.nvidia.com/t/357663 | 2026-07-29 |
| S-forum-170hx-spark | forum | Cheaper 1T VRAM via CMP 170HX + 4 Sparks — confirms DGX Spark has NO locked/hidden memory (unlike 170HX crypto cards); tcgen05/DSMEM/TMEM/TMA/multicast gap referenced from dgx-spark-playbooks; 8× 170HX benchmark 30 tg/s GLM5.2 4-bit (not GB10-specific) (Ria33, FlossingEnthusiast, alexander.korolev.germany) | https://forums.developer.nvidia.com/t/378348 | 2026-07-29 |

## Batch 75 forum sources (2026-08-17 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-fan-headless-boot | forum | DGX Spark (GB10) fans do not spin in headless boot mode — temp rises to ~60-70°C at idle; HDMI connected + reboot fixes; NVIDIA staff cannot reproduce; 2nd user reproduces on 1 of 2 identical ASUS GX10 units (display-hotplug dependent, unit-specific); driver 580.126.09, kernel 6.17.0-1008-nvidia, DGX OS 7.4.0 (jasonzhou_spk, aniculescu, josephbreda, solodu1116) | https://forums.developer.nvidia.com/t/361960 | 2026-02-28 |
| S-forum-suspend-fail | forum | nvidia-suspend.service fails on DGX Spark GB10 — nv_set_system_power_state WARNING (nv.c:4784), PCI PM suspend returns -5, s2idle suspend never completes; /sys/module/nvidia/parameters/ does not exist; nvidia-sleep.sh echo to /proc/driver/nvidia/suspend hits I/O error; driver 580.173.02, kernel 6.17.0-1029-nvidia (tsetjpc) | https://forums.developer.nvidia.com/t/380263 | 2026-08-15 |

## Batch 72 forum sources (2026-08-16 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
|| S-forum-spark-comfyui | forum | spark-comfyui — self-healing ComfyUI lifecycle manager for DGX Spark: one script (install/run/update/doctor/status/tune/service), SageAttention compiled for sm_121, GPU onnxruntime, UMA get_free_memory patch, NVFP4 kernel verification, TRITON_PTXAS_PATH fix, stuck-clock detection, backup/restore/reset, full containerization, MiniMax-H3 support, recipes (Turrican) | https://forums.developer.nvidia.com/t/376542 | 2026-07-11 |
|| S-forum-dsv4-0731-b12x-hang | forum | DeepSeek-V4-Flash-0731 b12x build hangs on 2× Spark during safetensors load at 92% — InstantTensor draft loader mod, fix-dspark-sm120-topk mod, upstream fastsafetensors PP-scoped draft load deadlock (vllm-project/vllm), sparkrun auto-determines NCCL config, INSTANTTENSOR_DRAFT_LOADER=instanttensor workaround (vincenzoa, frankyyyyy, hypermac.6502, Ama5u, dbsci) | https://forums.developer.nvidia.com/t/379184 | 2026-08-04 |

## Batch 44 forum sources (2026-07-31 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-xid31-yolo | forum | Repeated Xid 31 MMU faults during AMP-enabled YOLOv8s training on DGX Spark (GB10) — 5/5 AMP runs produce Xid 31 (ENGINE GRAPHICS GPC2, FAULT_PDE ACCESS_TYPE_VIRT_READ); CUDA_LAUNCH_BLOCKING=1 surfaces cuDNN CUDNN_STATUS_EXECUTION_FAILED at conv2d; FP16 matmul loop 3000s/208K iterations clean; AMP-disabled run 3.5h/7 epochs clean (limited evidence); NGC PyTorch 26.06-py3, driver 580.159.03, CUDA 13.3 user driver 610.43.02; telemetry ≤71°C/44W; GPC2 consistency suggests possible hardware/firmware (dall9) | https://forums.developer.nvidia.com/t/378529 | 2026-07-29 |

## Batch 45 forum sources (2026-07-31 ingest)

||| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-um-kernel-init | forum | Unified-memory kernel-init allocations (FlashInfer/DeepGEMM/NVFP4) consume ~50GB before weight loading starts on GB10 — disables vLLM auto-prefetch for large NVFP4 checkpoints; nvcr.io/nvidia/vllm:26.07-py3, ~118GB available drops to ~45GB after kernel/backend selection (rp_37716) | https://forums.developer.nvidia.com/t/378585 | 2026-07-30 |
| S-forum-vllm-2607-xgrammar | forum | nvcr.io/nvidia/vllm:26.07-py3 tool-calling 500 — xgrammar 0.2.0 missing normalize_tool_choice (added in 0.2.4); workaround Dockerfile pip install -U xgrammar + re-pin transformers==5.6.1; NVIDIA confirmed internal ticket; json_schema + guided_regex also fixed; Qwen3.6-35B-A3B-FP8 DeepGEMM layout.hpp:59 assertion (rp_37716, Neill/NVIDIA) | https://forums.developer.nvidia.com/t/378582 | 2026-07-30 |
| S-forum-qwen36-draft-train | forum | Training a personal draft model for Qwen3.6-35B-A3B on DGX Spark — ~50 tok/s sustained decode with MTP nst=3 on vLLM; community advice: use existing DFlash drafter (z-lab/Qwen3.6-35B-A3B-DFlash) over training custom DSpark, DSpark only marginally better (colizu2020, alexander.kachur) | https://forums.developer.nvidia.com/t/378611 | 2026-07-30 |
| S-forum-sm121-4bugs | forum | [SM121] 4 bugs causing `!!!` garbage output on NVFP4 models + gpt-oss-120B at 59 tok/s — root cause analysis: cutlass_fp4_supported() false positive, CutlassExpertsFp4 matches SM121, SupportsQuant missing on Qwen3.5, PTX+Marlin race; VLLM_MXFP4_BACKEND=marlin (not VLLM_NVFP4_GEMM_BACKEND); gpt-oss-120B 59 tok/s, Qwen3.5-35B MXFP4 59 tok/s, Qwen3.5-122B NVFP4 ~15 tok/s; corroboration: raphael.amorim confirms gpt-oss-120B 58-60 tok/s, Qwen3.5-35B FP8 52-55 tok/s, Qwen3.5-122B int4-AutoRound 28-29 tok/s (coolthor, raphael.amorim, learnerbs22) | https://forums.developer.nvidia.com/t/364009 | 2026-07-30 |
| S-forum-velogb10 | forum | veloGB10 — Rust-based inference engine optimized for GB10: custom kernels for "retail" GB10 chipset, TP=2 cluster support; Qwen3.6-27B-NVFP4-full ~40 tok/s single / ~45-50 tok/s 2×; Qwen3.6-35B-A3B-NVFP4 ~110 tok/s single / ~120+ 2×; Qwen3.6-9B ~80 tok/s single; pure NVFP4 (100% layers quantized); community feedback: 2× cluster numbers slower than vLLM for 27B dense, 35B MoE at parity with eugr vLLM at c=1 (stav_kats, jc2375, JW2026, robert287) | https://forums.developer.nvidia.com/t/377565 | 2026-07-20 |
| S-forum-cx7-pcie-power | forum | DGX Spark 2-node communication speed issue — CX-7 PCIe "insufficient power on the PCIe slot (27W)" dmesg on all 4 ports; iperf3 19.3 Gbits/sec with 6405 retries; ib_write_bw 111.60 Gb/sec (healthy); Qwen3.5-122B-FP8 TP=2 Ray worker fails after weight load (gloo connection closed); fix: NCCL all-reduce deadlock thread; concurrent request stall 20-30s before generation (ammarabbaxi13, mashie, aniculescu) | https://forums.developer.nvidia.com/t/378459 | 2026-07-30 |
| S-forum-hy3-1bit | forum | Hy3 1-bit GGUF on single DGX Spark — tight fit even at 1-bit, ~15 tok/s via llama.cpp, very intelligent but painfully slow; TurboQuant + llama.cpp fork pending to speed up and leave MTP headroom (branislav.djalic, phyo.arkarlwin) | https://forums.developer.nvidia.com/t/376870 | 2026-07-21 |
| S-forum-laguna-king | forum | Laguna-S-2.1 "new king?" — 63-post thread; mixed results: vr8vr8 40-50 tok/s c=1, 80-100 tok/s c=2; nuk3s 22.6 tok/s; Schampuswerner 19-24 tok/s decode, tool-eval 97/100 (vs Qwen 100/100), 86/100 hardmode (vs 91); robert287 updated NVFP4 45.5 t/s code / 27.2 structured; nuk3s temp sweep: updated quant flattens curve, +20% decode at temp 0.7; Poolside updated NVFP4 weights; PrismaQuant-GridBook 6-bit release pending; community consensus: not "new king," quality below Qwen3.6-35B-A3B for agentic (kyrylo.gorbachov, vr8vr8, nuk3s, Schampuswerner, robert287, mangosq, tenari, jwarner) | https://forums.developer.nvidia.com/t/377662 | 2026-07-30 |

## Batch 47 forum sources (2026-08-02 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-nemotron-2node | forum | Nemotron-3-Super-120B-A12B-NVFP4 on 2-node cluster — full vLLM recipe (TRITON_ATTN, cutlass MoE, fp8 KV, mamba_ssm_cache_dtype float32, fastsafetensors load), fp8 attention scaling-factor warnings (uncalibrated q/prob_scale 1.0), model pre-download required (launch-cluster.sh does not auto-download), 13.67-14.33 tok/s dual-node vs 15 tok/s single-node — dual-node slightly slower (elvis.dowson, eugr, mashie) | https://forums.developer.nvidia.com/t/378575 | 2026-07-30 |
| S-forum-dsv4-dspark-eugr | forum | DeepSeek-V4-Flash-DSpark on 2× Spark via Eugr's spark-vllm-docker — DSpark spec-decode recipe YAML, FlashInfer PR 3817 required, --load-format safetensors (or crash), 3 draft tokens beats 5 (71.63 vs 48.60 tok/s, 48.35% vs 27.65% acceptance), max_num_batched_tokens 10240 > 8192 for 3-draft, build-and-copy.sh -c uses IB (davidbarnesguildford, johndaly, Zambonilli, eugr_nv) | https://forums.developer.nvidia.com/t/376220 | 2026-07-09 |

## Batch 46 forum sources (2026-08-01 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-inkling-small-2x | forum | Inkling-Small-NVFP4 (276B/12B-active MoE) on 2× DGX Spark — NVFP4 fits 2 nodes; no FP8 KV cache support (BF16 KV only), caps context at ~300K; eugr spark-vllm-docker recipe + paged-KV mod; tool-calling parser bug (direct streaming emits tool-call markers as visible content), patched by ekkis via Codex; tool-eval-bench 76/100; vLLM blog: Inkling uses BF16 global attention, FP8 needs FlashAttention kernel mod; tonyd2wild BF16-KV 262K DSpark variant in progress; DSV4 uses less KV memory (PILCOTHINK, eugr_nv, 0rand, tonyd615, ekkis, adrianwild, jc2375) | https://forums.developer.nvidia.com/t/378645 | 2026-07-30 |
|| S-forum-inkling-small-disc | forum | Inkling Small announcement discussion — community testing plans; DSV4 GA expected soon with possible vision; DSV4 lower KV memory than Inkling-Small; Qwen3.5-122B FP8 as vision-capable alternative on dual Spark; eugr spark-vllm-docker experimental support; tool calls spitting raw markers (j-montoya, jwarner, CosmicRaisins, eugr_nv, thomas.developer1, user70634, peter.h177) | https://forums.developer.nvidia.com/t/378630 | 2026-07-30 |
|| S-forum-moe-lora-vllm | forum | LoRA training of MoE models (Qwen3.5-35B-A3B, Gemma-4-26B-A4B) + vLLM serving — Unsloth LoRA format mismatch with vLLM fused expert tensors; NVIDIA AutoModel/NeMo official MoE LoRA recipes (Gemma4-26B-A4B, Qwen3.5-35B-A3B) produce HF-compatible adapters servable via vLLM --enable-lora; Unsloth Studio OOM on these models (haidij, aniculescu/NVIDIA) | https://forums.developer.nvidia.com/t/366223 | 2026-04-10 |
|| S-forum-super-idol | forum | Super Idol Master: Multi-Agent 3D Character Asset Pipeline on DGX Spark — 9-agent system for 3D character production; AutoRemesher ARM64 patch (Geogram x86 assembly → portable C++); ComfyUI on DGX Spark for image generation; hybrid cloud-edge architecture (Windows control plane + Spark compute); application showcase, not LLM inference (fioricleto) | https://forums.developer.nvidia.com/t/377760 | 2026-07-22 |

## Batch 48 forum sources (2026-08-02 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-depfree-dashboard | forum | DGX-Spark-Dashboard — dependency-free monitoring dashboard built for single DGX Spark; FastAPI + vanilla HTML/CSS/JS, no DB/agent/CDN; ~190 MB image, ~42 MiB RAM, ~0.2% one core idle (measured on GB10); vs DCGM+Prometheus+Grafana ~600 MiB RAM, ~2.5 GB images (~14× mem, ~13× disk); demand-driven (no background collector); NVML for GPU data (not nvidia-smi — nvidia-smi polling is a performance killer); read-only /proc + Docker socket, non-root, cap_drop:ALL; single-node only (CX-7 ports not monitored) (angads25, mashie, elsaco) | https://forums.developer.nvidia.com/t/377085 | 2026-07-16 |

## Batch 49 forum sources (2026-08-03 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-cooler-temps | forum | Cooler GB10 Temps — 38-post thread, 2618 views; quantitative clock-cap A/B: 2000 MHz decode ≈0% loss, 55-69% power reduction, 8-22°C temp drop across 5+ independent users; cuBLAS SGEMM sweep shows -23% clock = -9% throughput (bandwidth-bound working set); systemd persistent clock-cap unit; KojiChou 3-node A/B (LLM ≈0% loss, diffusion +2.6-7.5%); whpthomas 12h quantization 0.6% loss at 1982 MHz; diffusion compute-bound ~12.5% loss (azampatti, whpthomas, KojiChou, ijontichy, paxren2020, g6.67300) | https://forums.developer.nvidia.com/t/372662 | 2026-06-15 |
| S-forum-grm32-sky | forum | OrionLLM/GRM-3.2-Sky (70 GB bf16, fits single Spark) — identified as Qwen3.5-35B/Ornith finetune; tool-eval-bench 86/100; tester finds worse than whpthomas/Ornith-1.0-35B-int4-AutoRound; no GB10-specific flags or configs (DannyTup, emX0r, stefan132) | https://forums.developer.nvidia.com/t/378777 | 2026-08-01 |

## Batch 50 forum sources (2026-08-03 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-dashboard-fw-stale | forum | DGX Dashboard stale firmware metadata — shows nvidia-firmware-580-580.159.03 as available update when 580.173.02 already installed; nvidia.ko loads gsp_tu10x.bin + gsp_ga10x.bin from 580.173.02 path; fwupdmgr vs apt package distinction (kafej666, amurnane123, sggin1, elsaco) | https://forums.developer.nvidia.com/t/378870 | 2026-08-01 |
| S-forum-dsv4-0731-caching | forum | vLLM prefix cache inconsistency on DeepSeek-V4-Flash-0731 on 2× Spark — sometimes cache hit (prefill 1-2s), sometimes complete miss (prefill minutes to tens of minutes); no deterministic cause identified; aidendle94/sparkrun-vllm-ds4-gb10 production-hybrid-1.1 image works on both dspark and 0731; 0731 slightly better than dspark in benchmarks (Sa0lence, dashtotherock, renek) | https://forums.developer.nvidia.com/t/378874 | 2026-08-01 |
| S-forum-4node-qrs812 | forum | 4-node DGX Spark Cluster with DSV4-Flash-0731-DSpark on QRS812 switch — TP=4, prefill ~2500 tok/s cold / decode ~90 tok/s C=1; KV cache hit prefill ~193K tok/s; C=6 decode ~40.4 tok/s/req; full-mesh RDMA latency matrix (write 2.93-3.49us, read 5.64-6.34us, send 2.55-3.44us); iperf2 ~105 Gbit/s, ib_write_bw 107.66 Gbit/s; NVFP4 DS-MLA KV cache dtype, MTP_NUM_TOKENS=3, vLLM 0.21.1rc1.dev339; mashie challenges: C12 2-node TP=2 230 vs 4-node TP=4 209 (jeffery2011.jc, mashie) | https://forums.developer.nvidia.com/t/378878 | 2026-08-01 |
| S-forum-laguna-yaml | forum | Laguna-S-2.1-NVFP4 YAML recipe for eugr's repo / sparkrun — TP=2, DFlash spec=15, CUTE_DSL_ARCH=sm_121a, MAX_JOBS=4; benchmark 50 reqs / 122.63 tok/s aggregate output / 268.58 tok/s total; DFlash acceptance 11.71%, acceptance length 2.76; per-position acceptance pos0=64.89% declining to pos14=0.78%; --kv-cache-memory=32449423258 override (davidbarnesguildford) | https://forums.developer.nvidia.com/t/378038 | 2026-07-24 |
| S-forum-powerstress | forum | partnerdiag PowerStress reproducibly hard-powers-off DGX Spark — external 1Hz thermal sampler caught zone0 88→97.8°C in 4s before power loss; zone2/zone4 sensor value swap anomaly persists across EC+SoC firmware updates (sensor handoff/mapping problem); post-firmware update (EC 0x03000302→0x03000508, SoC 0x0200980f→0x0200980b): box survives, returns error 082-000-1-020000600139 "temperature limits exceeded or thermal sensor broken/miscalibrated"; all other field tests pass; RMA approved; fieldiag ofed-scripts dependency bug corroborated; secure boot must be disabled for fieldiag install (digiegg, DannyTup, mashie, Neill/NVIDIA) | https://forums.developer.nvidia.com/t/377365 | 2026-07-18 |

## Batch 51 forum sources (2026-08-04 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-comfyui-triplany | forum | ComfyUI setup & patches for DGX Spark — UMA memory management fixes (double-VRAM, model eviction, memory spikes), comfy-aimdo 0.3.0 ARM compile fix, benchmarks (Z-Image/Flux2/LTX2.3/Wan2.2), LTX 2.3 22B NVFP4 ~12min/20s video (Triplany, TheAwakenOne, jd36) | https://forums.developer.nvidia.com/t/368344 | 2026-04-29 |
| S-forum-glm52-3x-aqlm | forum | GLM-5.2 (753B MoE, no prune) on 3× Spark TP=3 — NVFP4+AQLM hybrid checkpoint (272 GB, ~3.1 bits/param), virtual head padding 66 not 96, FlashInfer dispatch table mechanics, v3 kernel L1/L2 stream opts (+6.2%), v4 vision graft (MoonViT 16 heads not divisible by 3), benchmark methodology (t/s÷acceptance), 16GB swap mandatory, NCCL ≥2.30.7 hardcoded path (karol.spark, mashie) | https://forums.developer.nvidia.com/t/378150 | 2026-07-25 |

## Batch 52 forum sources (2026-08-04 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-fan-firmware | forum | DGX Spark low fan speed / high temps (62-post thread) — fan control entirely firmware, no PWM/BMC/OS override; swap exhaustion causes total lockup (corroborates S-forum-uvm-livelock, S-forum-llm-comfyui); ACPI thermal zones 92.8°C under load, fans ramp only at high-80s/90s threshold (nvidia3815, eugr, RazielAU, eggman, raphael.amorim) | https://forums.developer.nvidia.com/t/348760 | 2025-10-23 |
| S-forum-dsv4-0731-bench | forum | DeepSeek-V4-Flash-0731 official release — tool-eval-bench 87/100 (vLLM 0.25.2.dev0); DSML tool-call wrapper tag leaks to output at >60K context (regression in 0.26.1rc1.dev244); vLLM PR #49117 recovers missing wrapper but broken state persists at 150K; 4-config benchmark: TP4-seqs32 46.8-48.6 tok/s B1 (+33% vs TP2), C32 ~333-344; opencode_compat_proxy workaround (serapis, vedcsolution, Teason2026, penguinchang) | https://forums.developer.nvidia.com/t/378784 | 2026-07-31 |
| S-forum-earlyoom-config | forum | earlyoom on DGX Spark triggered too early during vLLM startup — default EARLYOOM_ARGS -s 80 kills container during Qwen3.5-122B-A10B init memory peak (lowering --gpu-memory-utilization to 0.75 doesn't help); fix: change -s 80 to -s 20 in /etc/default/earlyoom (trigger only when <20% swap left); sparkrun overwrites this on cluster config (helge) | https://forums.developer.nvidia.com/t/378934 | 2026-08-02 |

## Batch 53 forum sources (2026-08-05 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-jul31-wedge | forum | DGX Spark decode 107→45 tok/s after July 31 system update — power-controller wedge triggered by apt upgrade; AC power-cycle (wall unplug + wait) restores 84 tok/s; Qwen3.6-35B-A3B NVFP4 vLLM v0.25.0 (unicornxoxo2) | https://forums.developer.nvidia.com/t/379003 | 2026-08-03 |
|| S-forum-qlora-coding | forum | Nightly QLoRA fine-tuning Qwen3.6-35B-A3B on single DGX Spark — train bf16 / serve NVFP4 + --enable-lora hot-attach; flash-linear-attention 2.52× throughput win; NVFP4 has no gradient path; per_device_train_batch_size=1 GA=16; ~5.3 TFLOP/s effective; coding-agent SFT from Claude Code logs (jake.w.sims, emptysands) | https://forums.developer.nvidia.com/t/378311 | 2026-07-27 |

## Batch 54 forum sources (2026-08-05 ingest)

||| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-dsv4-0731-dspark-loader | forum | DeepSeek-V4-Flash-0731-DSpark on 2× Spark — DSpark draft loader weight mapping bug (shared_experts.w1/w3 → gate_up_proj missing → 12 tensors gone, invisible at INFO); fix: 32.7→55.4 tok/s (+69%), acceptance 25.7%→60.2%; SSE streaming trap (steps/s ≠ tok/s, use stream:false); draft quantization-config inheritance bug (vLLM PR #49133 — draft inherits target's NVFP4 config → ModelOptNvFp4FusedMoE on FP8 draft weights → acceptance collapse); 2× Spark TP=2 k=5 NVFP4 KV 1M ctx (tonyd615, srivatsa1) | https://forums.developer.nvidia.com/t/378824 | 2026-07-31 |
|| S-forum-macaron-v1-tall | forum | Macaron-V1-Tall (50B: 35B Qwen3.6-35B-A3B base + 4× 3.7B Rank-64 LoRA specialists) on single Spark — spark-vllm-docker vllm-node recipe, bf16, fp8 KV, 25-27 tok/s; MTP nst=3 gives 71.5% acceptance but only +2% throughput (41.93→42.79 tok/s); tool-eval-bench: base Qwen 90/100, full Macaron router 82/100 (routing sends most to L0 general chat); chat template fix (mods/fix-qwen3.6-chat-template); bf16 ~110 GB, no lower quants yet; OOM with LoRA specialists active (TheAwakenOne, jomark, emX0r, jetspark, 0rand) | https://forums.developer.nvidia.com/t/378436 | 2026-07-28 |
|| S-forum-qwen36-tp2-stall | forum | Qwen3.6-35B-A3B on 2× Spark TP=2 Ray — decode throughput collapses to 0.1-0.2 tok/s under concurrent requests; both GPUs at 105 GB memory; bf16 (not NVFP4), gpu_memory_utilization 0.85, max_num_seqs 8, flashinfer attention, prefix caching; vLLM logs show generation throughput 0.1-0.2 tok/s with KV cache usage <12%; single post, no replies (ammarabbaxi13) | https://forums.developer.nvidia.com/t/379105 | 2026-08-04 |
|| S-forum-acestep-v15-comfyui | forum | ACE-Step v1.5 on DGX Spark — install recipe (systemd, uv venv, vLLM backend for 5Hz-LM-4B lyrics model), measured numbers (memory ~15 GB, ~8s/generation fixed-cost not RTF), thinking:true = 2.7× timing variance; ACE-Step queues not batches; LTX-2.3 22B + ACE-Step full lip-synced music video in ComfyUI (32m16s end-to-end); --no-bf16-vae flag required for LTX-2.3 audio VAE (artfedderson, Turrican) | https://forums.developer.nvidia.com/t/378352 | 2026-07-28 |

## Batch 55 forum sources (2026-08-06 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-minimax-h3-comfyui | forum | MiniMax-H3 video generation on DGX Spark via ComfyUI — i2v/t2v/r2v 5s/0.2M timings (174s/143s/215s), ~235s at 768² with easycache+SageAttention KJ nodes, 432s for 10s video; models from Comfy-Org/MiniMax-H3; easycache + SageAttention KJ nodes setup (wxhpad, cx77, TheAwakenOne) | https://forums.developer.nvidia.com/t/379139 | 2026-08-04 |
| S-forum-cx7-idle-temp | forum | ConnectX-7 connection raises idle temperature ~10°C on DGX Spark (42→52°C) even with no load, fresh from boot — CX-7 chip powered off when no cable connected, adds ~17W heat per node when active (elvisnwh, mashie) | https://forums.developer.nvidia.com/t/379157 | 2026-08-04 |

## Batch 56 forum sources (2026-08-06 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-nondgx-os | forum | Non-DGX OS on DGX Spark — NVIDIA confirms DGX OS is officially supported, other OS possible but limited support; ACPI (not DT) means newer Linux distros work; Fedora 44 confirmed on GX10 (kernel 7.0.12-nv-1016.16, driver 595.84, CUDA 13.2); NixOS image (graham33/nixos-dgx-spark); NVIDIA-maintained kernels needed; Workbench/Field Diagnostics built for Ubuntu 24.04 only (NVES, hiroshiya, elsaco, sjug) | https://forums.developer.nvidia.com/t/379085 | 2026-08-04 |
| S-forum-dsv4-0731-ds4-cuda | forum | DeepSeek-V4-Flash-0731 on ds4 CUDA engine (Entrpi/ds4 fork v0.5.4) on single Spark — IQ2XXS quant, DSpark MTP k=2, 131K ctx, 40 tok/s decode, native C/CUDA binary; coder543: 1M ctx fits ~107GB with DS4_CUDA_NO_HBM_CACHE=1 + kv-disk-dir; full env vars documented (styles01, coder543) | https://forums.developer.nvidia.com/t/379192 | 2026-08-05 |
| S-forum-vllm-qemu | forum | vLLM QEMU emulation trap on GB10 — x86_64 Docker Hub vllm-openai images trigger QEMU on Grace CPU → 3.7 tok/s; CUDA 13 libcudart.so.13 not found (nested Python dir); pip install vllm replaces NVIDIA-optimized +nv PyTorch with generic build lacking sm_121 math kernels; Qwen2.5-Coder-32B-Instruct (rithinsundar87) | https://forums.developer.nvidia.com/t/378773 | 2026-07-31 |
| S-forum-crs812-4node | forum | MikroTik CRS812 DDQ for 4-node Spark cluster — disable auto-negotiation for 200G DAC, static RoCE IPs in netplan, MTU 9000 on switch + netplan, eugr docker image for vLLM; AI-generated RouterOS commands may look legit but fail; breakout mode 2x200G on qsfp56-dd ports (bhehe, urbanspr1nter) | https://forums.developer.nvidia.com/t/378431 | 2026-07-28 |
| S-forum-laguna-modelopt | forum | Laguna-S-2.1-ModelOpt-NVFP4-W4A4 on single Spark — 88/100 agent tool calls, 28 T/s; JasonW2025/Laguna-S-2.1-ModelOpt-NVFP4-W4A4-vllm HF repo; new ModelOpt W4A4 quant variant (JW2026) | https://forums.developer.nvidia.com/t/378501 | 2026-07-29 |

## Batch 57 forum sources (2026-08-07 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-cuda-single-ctx | forum | GB10 single-CUDA-context limitation — cuInit() returns CUDA_ERROR_NO_DEVICE (rc=100) for any second process while another holds a CUDA context, despite Compute Mode=Default; 20-60s teardown delay after process exit; vLLM sleep mode doesn't release context; Xid 119 interaction during teardown window; driver 580.159.03, CUDA 13, vLLM 0.25.x (tom450) | https://forums.developer.nvidia.com/t/379266 | 2026-08-05 |
| S-forum-cx7-27w-benign | forum | New DGX Spark setup — ConnectX-7 27W "insufficient power" boot warning on all 4 ports confirmed benign by NVIDIA staff (aniculescu); firmware inventory: EC 0x03000508, UEFI 0x02009b0b, CX7 28.45.4028, Samsung NVMe NXHB202Q; OTA 7.5.0, driver 580.173.02, kernel 6.17.0-1029-nvidia (james587) | https://forums.developer.nvidia.com/t/379261 | 2026-08-05 |
| S-forum-unsloth-docker | forum | Working Unsloth Docker recipe on DGX Spark — pytorch:25.10-py3 base, CUDA 13.0 nightly PyTorch, torchao dependency conflict fix (uninstall torchao), transformers+peft+datasets+trl+unsloth install, test_unsloth.py; NVIDIA playbook updated (Neurfer) | https://forums.developer.nvidia.com/t/350673 | 2025-11-08 |
| S-forum-vllm-fwdcompat | forum | vLLM 26.04-py3 "compatibility mode UNAVAILABLE" on driver 580.173.02 — forward-compat caps at CUDA 13.1 (590.48.01); 26.03+ tags need CUDA 13.2 / driver ≥595.58; no NGC tag both forward-compats and runs Qwen3.6 (model_type qwen3_5); 26.02 works but doesn't recognize Qwen3.6 (fmarcano, griffith.mark) | https://forums.developer.nvidia.com/t/379168 | 2026-08-04 |
| S-forum-thermal-freeze | forum | DGX Spark hard-freeze under sustained MiniMax-H3 inference — GPU 84°C, ACPI/SoC 93.1°C at 70-83W, no OOM/Xid/panic; PowerStress fails (MODS-020000610139: temp limits exceeded); clock-cap 2000MHz workaround corroborated; second user MiniMax-H3 at 58°C/15W (much cooler) shows unit-to-unit variation (tannerhaggerman, zc142365, sggin1) | https://forums.developer.nvidia.com/t/379195 | 2026-08-05 |

## Batch 58 forum sources (2026-08-07 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-sparkring | forum | SparkRing — 4× DGX Spark switchless ring inference for GLM-5.2 without Ethernet switch: custom SIRCL RDMA collective layer bypasses NCCL for inference collectives (TP4 all-reduce, DCP query/combine, all-gather, CUDA-graph-aware command rings); GLM-5.2 MXFP4-Experts-GPTQ 19-20 tok/s C1 / 50-63 tok/s C8 aggregate, 500K KV; MXFP8-NVFP4-NF3 hybrid 40-50 tok/s C4, 875K KV; EXL3 3.25bpw working with 1M KV room; SparkCache DCP4-sharded persistent NVMe KV cache; detailed bug report: GLM-5.2 indexer weights missing for 57/78 layers (top-k on uninitialized memory), launcher peer ordering XOR mismatch, VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1 missing, CUDA graph single-token lock (FujitsuPolycom, Terry01) | https://forums.developer.nvidia.com/t/378451 | 2026-07-31 |
| S-forum-dsv4-llamacpp-fan | forum | DeepSeek-V4-Flash-0731 UD-IQ2_M on HP ZGX (GB10) via llama.cpp — 524K ctx, 4 parallel, flash-attn on, --no-mmap, threads 10; tg32 16.2 tok/s, pp2048 390 tok/s, ttfr 4860ms; firmware update improved thermals to 71°C/75W under load with no shutdown; nvidia-smi -lgc 0,2000 clock cap corroborated in accessory thread (chrm) | https://forums.developer.nvidia.com/t/379276 | 2026-08-05 |
|| S-forum-cooling-fan | forum | Best $17 cooling fan for GB10 — USB case fan for ASUS Ascent GX10 bottom intake; nvidia-smi -lgc 0,2000 clock cap for <1% perf loss + 10°C temp drop; 3D-printed ducted cooling designs with 140mm filter (nathanpwhite, whpthomas, corbett_korbett) | https://forums.developer.nvidia.com/t/373199 | 2026-08-06 |

## Batch 59 forum sources (2026-08-08 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-kimi-k3-coder-reap | forum | Kimi K3 Coder REAP-320 MXFP4 on 8× GB10 — llama-bench decode 21-30 tok/s (tg1500), peak 35 tok/s, prefill 541-686 tok/s (pp2048); 32K context depth tested; REAP pruned variant "loops a lot" (quality issue); full K3 needs 16× GB10; same active expert count as full model (ciprianveg) | https://forums.developer.nvidia.com/t/378858 | 2026-08-01 |

## Batch 60 forum sources (2026-08-08 ingest)

|| ID | type | What it is | Reference | Date ||
|---|---|---|---|---|
|| S-forum-vllm-deepdive | forum | DGX Spark vLLM deep-dive blog posts — historical troubleshooting guide + technical report (sm_121 vs sm_100, CUTLASS/FlashInfer/Marlin, NVFP4/MXFP4/FP8 wiring, vLLM backend oracle, FP4 checkpoint-format rule, UMA OOM math, measured tok/s) (swesty) | https://forums.developer.nvidia.com/t/379391 | 2026-08-06 |
|| S-forum-dsv4-vision-plugin | forum | DeepSeek-V4-Flash-0731-vision on 2× Spark — FlyCockpit vLLM plugin (DeepEncoderV2 tower + 40 MB projector), DSpark wrapper-transparency fix (acceptance 1-15%→50-64%, ~40-50 tps), --limit-mm-per-prompt image:8, chat_template_kwargs thinking:false for images, tiles=2 token math 257/769/1281, vision quality assessment (co-le) | https://forums.developer.nvidia.com/t/379212 | 2026-08-05 |

## Batch 61 forum sources (2026-08-10 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-kernel-1029-rdma | forum | Kernel 6.17.0-1029-nvidia one-way RDMA regression on ASUS GX10 — ib_write_bw 13.2 Gb/s (DGX→GX10) vs 111.6 Gb/s (GX10→DGX); rollback to 6.17.0-1026-nvidia fixes to 111.7 Gb/s; 2 independent users confirm; TCP also asymmetric; CX7 firmware 28.45.4028, PCIe 5.0 x4 healthy (Claesbas, foogitiff) | https://forums.developer.nvidia.com/t/379303 | 2026-08-08 |
|| S-forum-vllm-snapshot | forum | vllm-snapshot plugin — fast model suspend/restore for vLLM sleep mode on GB10/SM121; snapshots built weight regions byte-for-byte to disk, restores via bulk cudaMemcpy ~1.6s / ~9s full swap vs 82s (safetensors) / 30min (instanttensor) for 26B NVFP4 MoE; vLLM sleep level-2 wake re-runs reload_weights (processing wall, not disk I/O); native vLLM plugin adds /suspend /restore endpoints; autosuspend + Compose depends_on for sequential boot; research prototype, only validated on GB10 (david.gareth.roberts) | https://forums.developer.nvidia.com/t/348862 | 2026-08-06 |
|| S-forum-dsv4-0731-gguf | forum | DeepSeek-V4-Flash-0731 GGUF (Unsloth release) — UD-Q8_K_XL 162GB lossless (7GB bigger than Q4); UD-IQ2_M runs on single Spark via llama-server (--ctx-size 262144 --parallel 2 --no-repack --cache-type-k f16 --cache-type-v f16 --reasoning off); MJPansa NVFP4 variant exists; MTP not yet compatible on community vLLM (vincenzoa, chriswalz86, anotheralvin, hammadsheikh) | https://forums.developer.nvidia.com/t/378829 | 2026-08-02 |
|| S-forum-dsv4-0731-sparkrun | forum | DeepSeek-V4-Flash-0731 DSpark via sparkrun — tonyd2wild DSpark 1M NVFP4 KV recipe packaged for sparkrun via brainchillz/sparkrun-dspark-registry; 58 tps agentic/coding on 2× Spark; derivative of existing tonyd2wild recipe (S-forum-dsv4-0731-dspark-loader) (david735) | https://forums.developer.nvidia.com/t/378892 | 2026-08-02 |

## Batch 62 forum sources (2026-08-10 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-crs804-8x | forum | CRS804-4DDQ for multi-Spark clusters — 8× Spark on CRS804 confirmed working; each GB10 QSFP56 port is 200GbE backed by 2× PCIe5 x4 links (~109Gbps per x4); QSFP56-DD breakout combines both x4 through one port for full 200G; must manually set port speed to 200Gbps; 2 breakout cables for 4× Spark; Naddod Q2Q56-400G-CU1 breakout DAC confirmed; QSFP-112 cables backwards-compatible but overkill (Spark port is QSFP-56); 10G ports for storage (jonasbridelu, jwarner, Patrick-ServeTheHome, elsaco, robert287, pdrayton, aostang, kevinpw, giles8, JW2026) | https://forums.developer.nvidia.com/t/364093 | 2026-08-10 |
| S-forum-dsv4-0731-dspark-llamacpp | forum | DeepSeek-V4-Flash-0731 DSpark on single GB10 via llama.cpp — first public dflash-format drafter (converted from ds4-fork); UD-IQ3_XXS 97GB + DSpark: 31 t/s code (accept ~50%, mean len 3.5/5), 20.2 t/s prose (accept ~25%); 16.6 t/s plain; --spec-draft-n-max 5 required; KLD quant ladder on 121GB: IQ3_XXS 0.24 vs Q3_K_XL 0.106, no intermediate viable; 256k ctx production: 25.3 t/s post-227k prefill, KV q8_0, 2 layers CPU experts, 9GB min avail; 151 t/s prefill (NEON-bound); CPU offload poisons speculation; llama.cpp copies weights to device buffers (no mmap serving) (GaelicThndr) | https://forums.developer.nvidia.com/t/379612 | 2026-08-08 |

## Batch 63 forum sources (2026-08-11 ingest)

||| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-clock-energy-sweep | forum | GPU clock energy-efficiency sweep on 2× DGX Spark TP=2 (DSV4-Flash-0731 FP8 + DSpark + prefix caching) — 17-point clock sweep (400–2400 MHz); decode bandwidth-bound (flat 47-51 tok/s from 1400-2400); best Wh/1M tokens band 1400-1800 MHz (~1350 vs 1688 uncapped = 25% better for ~3% speed loss); 2-node wall power 330W uncapped → 234W @1400; nvidia-smi accounts for only 12-27% of real GB10 draw; -lgc doesn't survive reboot (snaps to discrete steps); prefill is compute-bound (~14% penalty at 1400 vs uncapped); systemd persistent clock-cap unit (jetspark); stock cooling throttles at 2100MHz without repaste (arctic.gus); co-le corroborates with DSV4-Flash-0731 + DSpark at 2000-2150MHz (peter.h177, jetspark, arctic.gus, co-le) | https://forums.developer.nvidia.com/t/379389 | 2026-08-06 |
|| S-forum-lmcache-ipc-deadlock | forum | LMCache 0.5.3 MP mode + aidendle94 DS4F fork IPC deadlock on 4× Spark — LMCache server registers 170 KV layers for DS4F sparse-MLA hybrid KV, but vLLM hangs at "Wrapping 170 KV cache tensors for IPC" and times out after 300s; root cause: version gap (fork is vLLM 0.11.2.dev279 from 2025-10, LMCache 0.5.3 targets vLLM 0.18/0.20+); no LMCache version matches both fork's IPC surface AND DS4F hybrid KV; --disable-hybrid-kv-cache-manager breaks startup (KV demand 160GB > available); container needs --gpus all + --ipc host --shm-size 16g + cupy; Mooncake connector also unverified on this fork (mxjohnwong) | https://forums.developer.nvidia.com/t/379624 | 2026-08-09 |
|| S-forum-kat-coder-autoround | forum | KAT-Coder-V2.5-Dev-MTP-int4-AutoRound-SAR on DGX Spark — Spark AutoRound int4 quant of KAT Coder v2.5 Dev (Qwen3.6 MTP headers added); 85+ t/s accepted with MTP on Asus GB10; tool-eval 84/100 vs Ornith-1.0-35B-int4-AutoRound 87/100; OP finds Ornith superior for coding/planning; DannyTup benchmarks (bfcl/bigcodebench/ifevalcode) found it "pretty bad" and notes benchmark validation issues; azampatti plans to include in upcoming benchmark suite (SlopOps, whpthomas, DannyTup, azampatti) | https://forums.developer.nvidia.com/t/379611 | 2026-08-08 |

## Batch 64 forum sources (2026-08-11 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-xconfig-recovery | forum | nvidia-conf-xconfig.service recovery on DGX Spark FE — service is part of nvidia-conf-xconfig package, runs /usr/sbin/nvidia-conf-xconfig, RequiredBy=systemd-logind.service; missing service blocks GDM; /etc/apt/sources.list.d/spark.sources + GPG key lost after manual package deletion; extracted from System Recovery image; apt directly on Spark risks breaking kernel+driver pairing (hpcm, elsaco, Neill/NVIDIA) | https://forums.developer.nvidia.com/t/379141 | 2026-08-04 |
| S-forum-embed-rag | forum | Embedding + reranking models on single Spark for RAG under vLLM — --runner pooling flag; nomic-embed-text-v1.5 at gpu_memory_utilization 0.05, bge-reranker-base at 0.015; BAAI/bge-m3, NVIDIA Nemotron-3-Embed-8B/1B-BF16, Qwen 0.6B recommended; Qdrant + Open WebUI; chunk size 1200/overlap 200 (emX0r, ajvazan, joshua.dale.warner, jetspark, jrsphd, giraudremi92) | https://forums.developer.nvidia.com/t/379652 | 2026-08-09 |

## Batch 65 forum sources (2026-08-12 ingest)

|| ID | type | What it is | Reference | Date ||
|---|---|---|---|---|
| S-forum-spark-field-notes | forum | Cross-engine single-Spark benchmark field notes — same harness across Ollama and vLLM for 4 models (Qwen3-Coder-Next, Gemma-4-26B-A4B+MTP, GPT-OSS-120B, Qwen3-30B-A3B); Ollama does not batch (8x aggregate = single-stream), vLLM 289-313 tok/s at 8 concurrent; NVFP4 1.1 tok/s on vanilla vLLM (emulation fallback) vs 77.1 tok/s with FlashInfer-CUTLASS; SM121 lacks cvt.e2m1x2 (no FP4 compute, bandwidth saving only); MTP nst=2 beats 4 (acceptance decays 0.84→0.60→0.39→0.27); nvidia NVFP4 Gemma-4-26B-A4B 30.3 tok/s vs community Q4_K_M GGUF Ollama 49.6 — "obvious A/B makes NVFP4 look slower than it is" (ss121) | https://forums.developer.nvidia.com/t/379766 | 2026-08-10 |
| S-forum-glm52-8x-nvfp4 | forum | GLM-5.2 NVFP4 on 8× Spark TP=8 via eugr spark-vllm-docker — 25 tok/s decode, 256K context, tool-eval-bench v2.5.1 score 93; official nvidia/GLM-5.2-NVFP4 weights; recipe 8x-spark-cluster/glm-5.2-nvfp4.yaml worked right away (hypermac.6502) | https://forums.developer.nvidia.com/t/379765 | 2026-08-10 |
| S-forum-m3-nvfp4-4x-1m | forum | MiniMax-M3 NVFP4 on 4× Spark — 1M context (1,177,344-token KV pool via 4-bit packed nvfp4 KV), ~31 tok/s decode with EAGLE3, native vision + tool-calling; root-cause + fix for garbage output: NVFP4-Marlin MoE path drops SwiGLU-OAI activation params (gemm1_alpha 1.702 / gemm1_beta 1.0 arrive at Marlin kernel as defaults 1.0/0.0; clamp 7.0 threads through; same bug class as #46816/#47552 sibling on NVFP4 quant-config chain); 3-file param threading fix (config.py + nvfp4.py + modelopt.py); reasoning parser _looks_like_rendered_prompt guard for mm:think leak; enforce-eager mandatory (fp8-KV attention backend rejects nvfp4 KV + FULL cudagraph); NCCL ulimit nofile=1048576; --kv-cache-memory-bytes explicit; official nvidia/MiniMax-M3-NVFP4 (raw checkpoint, not community re-quants); vllm 0.17.2rc1.dev3669 nightly-dev-arm64-cu13.0.1-515d6e9 (baristankut) | https://forums.developer.nvidia.com/t/376979 | 2026-07-15 |

## Batch 66 forum sources (2026-08-12 ingest)

|| ID | type | What it is | Reference | Date ||
|---|---|---|---|---|
| S-forum-driver595 | forum | Driver 595.58.03 / CUDA 13.2 availability for DGX Spark GB10 — NVIDIA staff (aniculescu) confirms driver 595 not yet supported on DGX Spark; GB10 absent from supported devices list; CUDA 13.2 had issues, may jump to 13.3 (chrm, aniculescu, _cjg) | https://forums.developer.nvidia.com/t/364688 | 2026-03-25 |
| S-forum-fan-dpms | forum | DGX Spark fans stop when display blanks (DPMS off) or headless — box overheats; fans tied to display/USB power-draw state not thermal sensors; multiple independent confirmers (dmayer1, x1917x, sjug); workarounds: xset -dpms, VNC with active app, USB load ≥5W; NVIDIA staff (Neill) engaged; distinct from EC fan-curve regression (different mechanism — fan controller responds to SoC power draw, not thermal sensors); driver 580.159.03 (cold unit) vs 580.173.02 (hot unit) may be a factor | https://forums.developer.nvidia.com/t/378945 | 2026-08-03 |
| S-forum-trtllm-readout | forum | One-Forward-Pass Readout Engine for TensorRT-LLM on 3× DGX Spark — claims 1,014 tokens in 92.3 ms via non-autoregressive readout decoder on Nemotron-3-Nano-Omni-30B-A3B-NVFP4; sm_121, CUDA 13.0, TensorRT 10.14.1.48; self-described "vibe-coded" by non-expert; extraordinary unverified claim (lcoleman0422) | https://forums.developer.nvidia.com/t/379060 | 2026-08-03 |
|| S-forum-opengauntlet | forum | OpenGauntlet — 31 conversational LLMs benchmarked on single DGX Spark with TTS/STT; measured TTFT and tok/s at 512/2048/8192 prompt tokens across vLLM/sglang/llama.cpp; Gemma-4-26B-A4B NVFP4 variants 37-43 tok/s, Qwen3.6-35B-A3B 28 tok/s, Nemotron-3-Super-120B 13.5 tok/s; quality ELO ratings; aarch64 compatibility notes (AI_D3veloper) | https://forums.developer.nvidia.com/t/379037 | 2026-08-03 |

## Batch 67 forum sources (2026-08-13 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-h3-solattn | forum | DGX Spark One-Click Deploy for MiniMax H3 — 12 ComfyUI workflows + Sol-Attention (keys-heretic) Blackwell sm_121 adaptation; ~6 min for 5s 720p video, ~165 GB weights, install wizard; alexlu0912/dgxspark_comfyui_minimax_h3 (alexlu0912) | https://forums.developer.nvidia.com/t/379894 | 2026-08-12 |
|| S-forum-dragonscale | forum | Everything about benching — tool-eval-bench v2.5.0 scores 5-8 pts lower than v2.0.1 (version comparability issue); DragonScale agentic-coding bench (deterministic rubric, no LLM judge, builds Flappy Bird game); DSV4-Flash-0731 cloud 80, kimi-k3 84, Ling-3.0-Flash 66; DSV4-Flash-0731 local 2× Spark 85 hardmode (0rand, karol.spark, jetspark, serapis, azampatti) | https://forums.developer.nvidia.com/t/379373 | 2026-08-06 |
|| S-forum-openpangu | forum | openPangu-2.0-Flash (100.14B MoE, 92B params, 6B active, 256 experts, 512K ctx) — BF16 only on HF (184 GB, doesn't fit 1× Spark); Q4_K_M GGUF 52 GB exists (mrexodia/openPangu-2.0-Flash-GGUF) with fused DSA/mHC kernels; MTP-on-UMA analysis: speculation helps more on bandwidth-bound UMA, expert union width grows with draft depth on MoE, low depth wins high depth loses (GaelicThndr) | https://forums.developer.nvidia.com/t/378654 | 2026-07-31 |
|| S-forum-glm52-200k-4x | forum | GLM-5.2 (unpruned) QuantTrio Int4-Int8Mix on 4× Spark TP=4 — 27 tok/s single / 52.5 tok/s @c4, 200K ctx, MTP k=4, fp8_ds_mla KV, FLASHMLA_SPARSE+DSA; reproduction of tonyd2wild recipe; preset-PR drift gotcha (VLLM_APPLY_PRESET_PRS silently merges rebased PR branches → fp8_ds_mla page-padding crash); agent concurrency monitoring: queue depth > token flow > acceptance rate (baristankut, WanPiCo) | https://forums.developer.nvidia.com/t/377879 | 2026-07-23 |

## Batch 68 forum sources (2026-08-14 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-power-mgmt | forum | DGX Spark power management for multi-node clusters — no sleep/suspend viable for automated wake; smart plug + Auto Boot BIOS is the only working pattern; 4-node cluster idles 238W (no switch) / 260W (with CRS504), 800W+ during inference; 200W savings from clock cap at 1400 MHz (~5-10% speed loss); stopping services + CX7 + restart takes same time as cold start (CosmicRaisins, jetspark, mashie, peter.h177) | https://forums.developer.nvidia.com/t/379778 | 2026-08-11 |
| S-forum-nemotron35-lightning | forum | NVIDIA Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash on single DGX Spark — vLLM 0.27.1 ARM64, hybrid Mamba-2+MoE+Attention, 3B active, ~21 GiB weights; 78.5 tok/s target-only, 90.7 tok/s with DSpark (+15.6%, 53% acceptance, 1.59 accepted/draft); tool-eval 77/100 (DSpark 80/100) vs Qwen3.6-35B 100/100; 120+ tok/s decode reported (styles01 sparkrun-recipes YAML); 1M ctx, fp8 KV, Marlin MoE, mamba flashinfer (Schampuswerner, styles01, 0rand, gaborm) | https://forums.developer.nvidia.com/t/379832 | 2026-08-11 |
| S-forum-dsv4-qwen-vision | forum | DeepSeek V4 Flash + Qwen3.5-9B vision co-hosting on 2× DGX Spark — DSV4 ~200 GB across 2 nodes + Qwen vision ~30 GB = 230 GB total; gpu_memory_utilization 0.75 squeezes KV cache room; GitHub repo with scripts; stu.miller uses 3rd Spark with Qwen3.5-122B for vision; PILCOTHINK Pilco-mmbridge text-to-multimodal bridge + Qwen3.5-9B-MixedInt4-AutoRound quant (aditya1503, stu.miller, PILCOTHINK) | https://forums.developer.nvidia.com/t/379791 | 2026-08-11 |
| S-forum-dsv4-0731-llamacpp-bugzy | forum | DeepSeek-V4-Flash-0731 on single DGX Spark via mainline llama.cpp — UD-IQ2_M, 131K ctx/slot (524288÷4 parallel), 19.7 tok/s single-stream, 52 tok/s @ 4 concurrent; IQ3_XXS hits reproducible throughput dip at c=4 (47.6→26.3 tok/s) absent in IQ2_M; 96% correctness on 5-category eval; KV q8_0 garbles output on this model; coder543 ds4-on-spark comparison: DSpark 0% accept in continuous-batch path at target ctx, ~22 vs 19.7 tok/s single-stream; ds4 env vars for 1M ctx (DS4_CUDA_NO_HBM_CACHE=1, kv-disk-dir); llama.cpp can't represent DSV4 native mixed fp8/fp4 KV → 16-bit or quality-loss 8-bit (bugzy, coder543) | https://forums.developer.nvidia.com/t/379129 | 2026-08-04 |

## Batch 69 forum sources (2026-08-14 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
| S-forum-wifi-mesh | forum | MT7925e WiFi mesh network incompatibility + CX-7 DAC EMI on DGX Spark — WiFi auth loop (8+ retries to connect) specific to mesh WiFi equipment, works fine with simple router; cheap/unshielded DAC cable in CX-7 ports causes similar WiFi auth failures (EMI); fix: move DAC to outermost port or use recommended shielded DAC; both issues on 2× new DGX Sparks (simply_today, hoesing) | https://forums.developer.nvidia.com/t/379782 | 2026-08-11 |
| S-forum-idle-lockup | forum | DGX Spark silent idle hard lockup — LPI-3 deep-idle wake failure; distinct from OOM/thermal/wedge freezes; ASUS GX10, DGX OS 7.5.0, driver 580.173.02, kernel 6.17.0-1029-nvidia; ~97% memory free, zero GPU workload, zero forensic trace (no panic/OOM/Xid/hung_task); SoC descending into deepest idle state (LPI-3) at moment of freeze; only happens at idle, never under load; 7+ occurrences across 2 users; NVIDIA staff (aniculescu) confirms OOM freeze is known issue; embedding models via transformers show UMA-specific memory leak absent on x86 (luis.poveda9321, icoicqico123, aniculescu) | https://forums.developer.nvidia.com/t/358951 | 2026-01-28 |
|| S-forum-sparkup | forum | Sparkup — Ansible playbook provisioning DGX Spark from fresh DGX OS install to working training box; includes spbm firmware module for whole-system power telemetry (nvidia-smi sees only GPU rail, ~half wall draw); Docker+NVIDIA runtime, users+SSH keys, Prometheus+Grafana observability with 1-year retention (vladtemian) | https://forums.developer.nvidia.com/t/379218 | 2026-08-05 |

## Batch 71 forum sources (2026-08-15 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-gsp-reboot-jul2026 | forum | GB10 spontaneous reboots after July 2026 firmware update on Dell Pro Max — EC 0x02000b00, SBIOS 5.36_4.0.0, driver 580.173.02; GSP health check fail (kgspHealthCheck_TU102), NVRM assert flood at gpu_user_shared_data.c:373, Xid 120 GSP task exception (supervisor timer interrupt); sbsa_gwdt watchdog action=1 panics after GPU wedges → auto-reboot; 3 crashes at ~2h intervals; root cause: fwupd capsules applied on warm reboots without AC power disconnect; fix: full AC power disconnect → 24+ hours clean; fwupd-refresh.timer exonerated; DGX OS default sbsa_gwdt action=1 (nvidia-sbsa-gwdt-options package) (nathanviveiros, dgxspark, elsaco) | https://forums.developer.nvidia.com/t/379959 | 2026-08-12 |
| S-forum-muse-glimmer | forum | Meta Muse Glimmer 30B dense model on DGX Spark — llama.cpp UD-Q6_K_XL + DFlash: 44.6 tok/s @ d0, 35.0 @ d4096, 26.7 @ d8192 (tg128 c1); coder543: 13 tok/s no DFlash, 25 prose / 35 code with DFlash, prefill 1000→725 tok/s with DFlash; vLLM NVFP4 (Preyazz) + DFlash 18.65 tok/s agg vs BF16 7.72 (2.42×); vLLM DFlash broken (DFlashMuseGlimmerAssistantModel missing); tool-calling BFCL 10-12% (multi-tool failures across vLLM+SGLang); controlled A/B: 20/20 single-tool, DFlash 4.08× speedup; model sensitive to reasoning truncation; ~31 GiB memory (emX0r, DannyTup, 0rand, coder543, gaborm, contact.cgp.mgm) | https://forums.developer.nvidia.com/t/379722 | 2026-08-10 |
|| S-forum-pilco-mmbridge | forum | Pilco-mmbridge — text-to-multimodal bridge (text-only LLM + vision model without fine-tuning) on 2× DGX Spark; DeepSeek-V4-Flash-0731 + Qwen3.5-9B vision; DSpark spec decode; memory tuning: initial recipe OOMs → max-num-seqs 10→5, max-num-batched-tokens 4096→2048, max-model-len 1M→524K; final stable recipe: --kv-cache-memory-bytes 11859478446, gpu_memory_utilization 0.75; vision model Qwen3.5-9B-quantized.w4a16 at util 0.05, --kv-cache-memory-bytes 367001600; image persistence bug fixed; FE-only tested (PILCOTHINK) | https://forums.developer.nvidia.com/t/378850 | 2026-08-01 |

## Batch 73 forum sources (2026-08-16 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-muse-glimmer-nvfp4-w4a4 | forum | Inferact/Muse-Glimmer-30B-NVFP4-W4A4 benchmark on DGX Spark with vLLM — 52.55 tok/s text generation; Spark Arena Benchmark submission (kuscsik) | https://forums.developer.nvidia.com/t/380229 | 2026-08-14 |
| S-forum-energy-telemetry | forum | GB10 Grace CPU energy telemetry full audit + peer-reviewed paper — DCGM field 156 (cumulative mJ) works for GPU rail; no CPU energy interface works (SCMI, I2C, hwmon, powercap all empty); SPBM NVDA8800:00 fails on GX10 due to MTKW9000 memory conflict (ACPI resource overlap 0x05170000); spark_hwmon loads cleanly on Acer GN100 (no conflict); nvidia-smi --query-gpu=energy.consumption returns "Field not valid"; DCGM fields 1130/1132/1133 return 0.000; July EC firmware update does NOT fix energy gap; arXiv:2605.27599 (LOCO 2026 accepted); NVIDIA staff (aniculescu) says should be fixed in July updates (deepak.panigrahy03, aniculescu) | https://forums.developer.nvidia.com/t/371614 | 2026-08-14 |
|| S-forum-asm2464pd-replug | forum | ASM2464PD USB4 NVMe enclosure falls back to USB 2.0 (480 Mbps) on ASUS GX10 after every boot — soft-replug script automates the fix using reverse-engineered CPU reset sequence (cyrozap/usb-to-pcie-re: e8 50 + 13×00 register write); multiple users confirm issue since Oct 2025 (JW2026, paulsc.liu, gaborm, vedcsolution) | https://forums.developer.nvidia.com/t/380009 | 2026-08-14 |
|| S-forum-4xdgx-glm52 | forum | 4× DGX Spark model selection discussion — durable GLM-5.2 data point: rdoiron running ciprianveg's gb10-glm-5.2:v18-vision stack with DCP2 + decode-aware prefill + NVFP4 KV @ 300K context, prefill 700+ tok/s, decode 25 tok/s; karol.spark links GLM-5.2 4× recipes (0xdfi adaptive MTP, tonyd2wild QuantTrio 200K); rest of thread is model preference debate (GLM-5.2 vs DSv4-Flash) (ivr718, karol.spark, rdoiron, ciprianveg) | https://forums.developer.nvidia.com/t/379295 | 2026-08-06 |

## Batch 74 forum sources (2026-08-17 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-fe-thermal-rma | forum | 2× DGX Spark FE silent hard-locks under sustained DSV4-Flash-0731 inference — fieldiag PowerStress FAIL (020000600139) on both units at stock, latest firmware; GPU clock cap 2100 MHz + CPU freq cap 2400 MHz stabilizes; 12 lock events, 3× reproduced per unit across fieldiag 1.0.9 + 2.0.4; RMA approved ~48h; fieldiag 2.0.4 install gotchas (ofed-scripts dep, CX7Stress link-down, 1.0.9 launcher leftover); 120s hw watchdog fails 3/4 events; one unit hard-locked mid-PowerStress under MODS-only (no OS) (tniccum) | https://forums.developer.nvidia.com/t/380238 | 2026-08-14 |

## Batch 77 forum sources (2026-08-18 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-cx7-promisc | forum | ConnectX-7 promiscuous mode failure on DGX Spark — rte_eth_promiscuous_enable() and `ip link set promisc on` return success but hardware does not enter promisc mode (no PROMISC flag, non-local MAC packets not received); PSID NVD0000000087, CX7 part cx7_P4242_HORIZON_PK_Ax, FW 28.45.4028; DPI system use case (power_tang999) | https://forums.developer.nvidia.com/t/380361 | 2026-08-17 |
| S-forum-mediallmproxy | forum | MediaLLMProxy — OpenAI-compatible vision bridge for text-only LLMs on DGX Spark; Qwen 3.0 3B VL backup model on llama.cpp on one Spark, Qwen 3.6 35B A3B VL as main vision on Mac; DeepSeek-OCR-2 explored (weak for non-OCR tasks); production-deployed media bridge (0rand) | https://forums.developer.nvidia.com/t/380317 | 2026-08-16 |
| S-forum-hdmi-hotplug-ab | forum | ASUS GX10 HDMI hot-plug A/B test — controlled display-state experiment on 2 identical units: HDMI connected + X11 session = 36-37°C; HDMI physically unplugged (session stays active) = monotonic rise to 45°C GPU / 47.9°C ACPI in 5 min (P8, 3.48-3.71 W); headless control unit (no display, GDM greeter) stays cool at 34-35°C; display-hotplug state linked to thermal/fan behavior independently of compute load; nvidia-smi fan N/A, no PWM nodes; per-unit EC/fan hardware difference suspected (solodu1116, x1917x, ajvazan) | https://forums.developer.nvidia.com/t/380282 | 2026-08-15 |
|| S-forum-usbc-dp-hpd | forum | USB-C DisplayPort (DFP-1 to DFP-4) not detected after boot on MSI EdgeXpert GB10 unless monitor cable physically replugged — /sys/class/typec/ empty, /sys/bus/thunderbolt/ empty; xrandr/nvidia-settings re-probe fails; only physical replug triggers EDID read; sink-detection window closes early in boot; 2nd confirmer on Lenovo Thinkstation PGX (native USB-C, no adapter); HDMI-0 works at boot; driver 580.173.02, kernel 6.17.0-1029-nvidia (riccardo1981, helge, Mkei88) | https://forums.developer.nvidia.com/t/380326 | 2026-08-16 |

## Batch 78 forum sources (2026-08-19 ingest)

||| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-qwen38-27b-mixedint4 | forum | Qwen3.8-27B-MixedInt4-AutoRound for single DGX Spark — PILCOTHINK mixed 4-bit quant (sensitive layers FP8/FP16, vision unquantized), 20.8 GB; vLLM recipe with MTP nst=3 + fp8 KV, 1.01M max ctx, 2.56M-token KV pool (2.54× concurrency); MMLU recovery 99.32% (83.49→82.92%, -0.57pp); llama-benchy 21.86 tok/s decode d0 / 17.04 @ d4096 / 17.82 @ d8192, prefill 828-877 tok/s; tool-eval-bench 91/100 normal / 92/100 hardmode v2.5.1; co-le 35-40 tps on 2× Spark; SlopOps SAR variant 88/100 hardmode v2.1.0, MTP nst=3 15.08 tok/s / nst=4 15.67 tok/s; 0rand DSpark 28-35 t/s 8-bit single Spark; dean.grande 27B planner + 35B coder dual-role; stu.miller dense 27B slow vs MoE 35B (PILCOTHINK, stu.miller, co-le, SlopOps, 0rand, dean.grande, nekoseraph) | https://forums.developer.nvidia.com/t/380248 | 2026-08-15 |
|| S-forum-docker-vs-k8s | forum | Docker Compose vs Kubernetes for DGX Spark orchestration — community consensus: spark-vllm-docker + sparkrun sufficient for 2-4 Sparks, k8s overkill unless integrating larger ecosystem; bugsareyummy dropped Rancher k8s to recover RAM for vLLM on UMA; ArgoCD GitOps for 4×2 sparks; sparkrun supports multiple cluster configs; minikube/k3s viable with minimal overhead (ngarrison, Zambonilli, bugsareyummy, tudor5, james.brooke, mashie) | https://forums.developer.nvidia.com/t/372636 | 2026-06-08 |

## Batch 79 forum sources (2026-08-19 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-glm52-sparkrun-4x | forum | GLM-5.2 QuantTrio Int4-Int8Mix on 4× DGX Spark via sparkrun recipe — davedgd/sparkrun-glm52-4x-spark repo; uses ciprianveg Docker image; supports baseten/GLM-5.2-Vision-NVFP4 vision tower; Adaptive MTP; tool-eval-bench v2.5.1.dev31: 86/100 (53 pass, 12 partial, 4 fail); llama-benchy pp2048 556 tok/s, tg32 22.17 tok/s; AIME25 90% (30/30, 57 tok/s avg, 290K tokens); engine vLLM 0.11.2.dev280+gilded.gnosis.v18; 1M max context (davedgd, stu.miller, robert287) | https://forums.developer.nvidia.com/t/380448 | 2026-08-17 |
|| S-forum-kexaone-236b | forum | K-EXAONE-236B-A23B (LG AI Research 237B MoE, 23B active) on single DGX Spark via ds4 engine — mixed-quant GGUF (IQ2_XXS+Q3_K+Q4_K+Q8 per-tensor-precision); 441.63 GiB BF16 → 85.56 GiB (5.16×); full 262,144-token context on one GB10 at 103.95 GiB resident; LLLG sliding-window schedule = 48 KiB/token KV (12/48 layers hold full context); MTP block (blk.48) executes but net loss on this HW; ds4-server with sm_121 build; OpenAI-compatible 16/16 validation checks; decode 10.51 tok/s @1.4K tokens → 5.42 tok/s @16K; prefill 53→42 tok/s; multi-turn prefix-resume divergence workaround (re-tokenization mismatch); Baekpica/K-EXAONE-236B-A23B-Mixed-Quant-GGUF (Baekpica, OllieOllie) | https://forums.developer.nvidia.com/t/379577 | 2026-08-08 |

## Batch 80 forum sources (2026-08-20 ingest)

|| ID | type | What it is | Reference | Date |
||---|---|---|---|---|
|| S-forum-dsv4-agent-serving | forum | Agent-serving on 2× DGX Spark with DSV4-Flash-0731 — corrected A/B: --max-num-batched-tokens is a fairness lever (not just jitter); mnbt=2048 gives 2.9× decode share during long prefill vs 8192 (5.0% vs 1.7%); ongoing stream gets ~1.9 tok/s vs ~0.6 tok/s; prefill cost -7.7% (1462 vs 1584 tok/s); 3 instrument faults corrected (SSE events ≠ tokens, no real prefill window, cold reference); 0rand suggests prefill batch size influences MTP acceptance (svangenstudios, 0rand) | https://forums.developer.nvidia.com/t/378890 | 2026-08-02 |
|| S-forum-gx10-fw-recovery | forum | ASUS GX10 firmware recovery from bricked state after interrupted apt update — firmware flash cycles through SoC→BIOS→EC resets over minutes, looks hung, power-cycling mid-flash bricks it; power-drain reset (unplug + hold power 60s) revived 1 of 2 units; official recovery image black-screens but stock Ubuntu 24.04 arm64 live USB boots; recovery: mount internal EFI partition, stage .cap capsule, arm OsIndications, reboot to flash; .cap uses OS-driven capsule-on-disk not in-BIOS menu; NVIDIA staff (Neill) confirms multi-reset flash is expected (aquaponicCowboy, Neill) | https://forums.developer.nvidia.com/t/372063 | 2026-06-02 |

## Batch 81 forum sources (2026-08-20 ingest)

||| ID | type | What it is | Reference | Date ||
||---|---|---|---|---|
|| S-forum-qwen38-nvfp4-vs-fp8 | forum | Qwen3.8-27B NVFP4 vs FP8 on single DGX Spark — vLLM 0.27.1, 16 concurrent; Unsloth Dynamic NVFP4 (4-bit MLP + 8-bit attn + FP8 KV) 30-34% faster than Qwen official FP8 (block 128, e4m3) across all workloads; NVFP4 87.91-134.41 tok/s (peak 128-144) vs FP8 65.58-104.44; model 23.4 GB vs 30.9 GB; KV ~5.6% vs ~15%; single-stream ~10-12 tok/s (racerdude); also tested on Jetson Thor with similar results; FP8 recipe with MTP nst=2, fp8 KV, lazy safetensors, CUTE_DSL_ARCH=sm_121a, VLLM_USE_DEEP_GEMM=1 (shahizat, racerdude) | https://forums.developer.nvidia.com/t/380258 | 2026-08-15 |
|| S-forum-ds4f-qwen38-orchestration | forum | Multi-model orchestration on 3+ DGX Sparks — DS4F on 2 Sparks as main + Qwen3.6-35B-A3B on 3rd for subagents + Qwen3.8-27B for code review/vision; Qwen3.8-27B too talkative for Hermes limits, Qwen3.6-27B better auxiliary; stu.miller uses 122B on 3rd Spark for coding/vision instead of 35B; "virtual MoE" pattern: DS4F as architect + Qwen 27B as reviewer (ajvazan, stu.miller, Hunlx, Ama5u) | https://forums.developer.nvidia.com/t/380426 | 2026-08-17 |
|| S-forum-uefi-capsule-password | forum | DGX Spark Capsule Update blocked by UEFI Administrator Password never set — fwupdmgr upgrade (EC 0x03000302→0x03000508, UEFI SoC 0x0200980f→0x02009b0b) completes but reboot stops at "Capsule Update / Enter Admin Password" blue screen; user never configured UEFI admin password; 3-cycle behavior (3×3 failed attempts → normal login); NVIDIA staff (aniculescu) confirms no reset/clear method, recommends RMA; corroborates S-forum-opal-uefi password corruption (burnsy56, aniculescu) | https://forums.developer.nvidia.com/t/380503 | 2026-08-18 |
|| S-forum-dsv4-humaneval | forum | DeepSeek-V4-Flash-0731 local vLLM vs cloud HumanEval — local 2× Spark TP2 (eugr/spark-vllm-b12x) 96.3% pass@1 vs OpenRouter 95.1% (temp 1.0, top_p 0.95, reasoning=max); serving config: max_model_len 1,048,576, max_num_batched_tokens 4096, max_num_seqs 6, DSpark k5, prefix-cache retention 4096; 0 errors/timeouts; local quant matches or exceeds cloud quality (florianbrede, 0rand, JW2026) | https://forums.developer.nvidia.com/t/379560 | 2026-08-07 |

## Batch 83 forum sources (2026-08-21 ingest)

|| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-qwen38-27b-vllm-mtp | forum | Qwen3.8-27B-NVFP4 on single DGX Spark — comprehensive vLLM+MTP measurements, 26-post thread, 9K views: working recipe (vllm serve unsloth/Qwen3.8-27B-NVFP4, gpu-mem-util 0.45, max-model-len 262144, MTP k=5, chunked prefill, prefix caching); tokenizer truncation bug (max_length 2048, now fixed by Unsloth); YaRN 1M context via hf-overrides (rope_type yarn, factor 4.0, text_config); architecture identical to Qwen3.6-27B (64 layers, hidden 5120, 24 heads/4 KV heads, 48 linear_attention + 16 full_attention, MLP NVFP4 + attn FP8 + vision bf16); KV cost 37,169 bytes/token; decode 24-26 tok/s with MTP k=5, 11.4 tok/s without MTP; 2-node cluster ~37 tok/s; param count analysis (27.78B, not 20B — U8 packing trap); styles01 recipe (drowzeys GB10 image, MTP n=3, 39 tok/s decode, 98 tok/s aggregate @ 4 concurrent); voktolom RTX 5090 comparison (NVFP4 vs PrismaAQUA 5.5-bit, tool-eval 90.7 vs 87.8) (helge, styles01, voktolom, mecworks_nvidia, jbourny, jomark, datltq) | https://forums.developer.nvidia.com/t/380244 | 2026-08-15 |
| S-forum-qwen38-nemotron-bench | forum | DGX Spark (GB10) model selection: Qwen3.8-27B 4 tok/s to Nemotron 3 Nano 56 tok/s — comprehensive single-Spark benchmark across 3 model generations, 3 precisions, 2 engines, 4 decode strategies. Qwen3.8-27B BF16 4.09 tok/s (vLLM) / 4.24 (SGLang), FP8 6.92-6.96, FP8+DSPARK 9.77, FP8+MTP k=3 12.24, NVFP4+MTP k=3 17.1, NVFP4+MTP k=5 18.4; Nemotron 3 Nano 30B-A3B NVFP4 55-56 tok/s (no spec decode); TTFT: FP8 no-spec 0.19s optimal, MTP 8.5-11s; bandwidth model: effective ~202 GB/s (74% of 273); cold start 290s (Nemotron) vs 7min (Qwen); memory: Nemotron 19.4GB weights, ~115GB resident, 256K ctx; vLLM MTP TTFT penalty finding; FP8 KV in hybrid models (GDN/Mamba) barely saves memory; Nemotron 3 recipe: vllm/vllm-openai:v0.27.1-aarch64, kv-cache-dtype fp8, reasoning-parser nemotron_v3, tool-call-parser qwen3_coder, gpu-mem-util 0.90, max-model-len 262144 (reborn.li.rl) | https://forums.developer.nvidia.com/t/380700 | 2026-08-20 |
| S-forum-vllm-longlived | forum | Playbook for keeping vLLM up as a long-lived service on GB10 (PR #104) — --gpu-memory-utilization is spending system RAM not just VRAM: at 0.75 on 121.7 GB, vLLM reserves ~91 GB at startup (53.8 GB KV cache sized for 31 concurrent at 131k context); idle cache measured 0.0% idle, 0.5% under load on single-user box; vLLM logs Available KV cache memory: 53.84 GiB / GPU KV cache size: 4,073,445 tokens / Maximum concurrency for 131,072 tokens: 31.08x (fede5) | https://forums.developer.nvidia.com/t/380721 | 2026-08-20 |
| S-forum-dsv4-nvfp4-416-kv | forum | Native 416-byte NVFP4 KV cache + DSpark for DeepSeek-V4-Flash-0731 on 2× DGX Spark — native 416-byte sparse-MLA KV-cache record (256B packed E2M1 + 32B E4M3 scales + 128B BF16 RoPE) vs padded 584-byte FP8 KV; 14% more KV capacity; fused cache writers + native FlashInfer sparse-attention reader; MAX_MODEL_LEN=1048576, MAX_NUM_SEQS=4, MTP_NUM_TOKENS=5, GPU_MEMORY_UTILIZATION=0.835; KV cache 17.66 GiB / 2,779,464 tokens / 2.65x concurrency at 1M ctx; DSpark fixes: stable state across requests, mixed prompt/decode batches, ragged rejection trimming; repo coolbho3k/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark branch feature/nvfp4-416-kv-cache (emihuang) | https://forums.developer.nvidia.com/t/379788 | 2026-08-11 |

## Batch 84 forum sources (2026-08-22 ingest)

| ID | type | What it is | Reference | Date |
|---|---|---|---|---|
| S-forum-prismaaqua | forum | PrismaQuant AQUA (activation-quantization-awareness) + Qwen3.8-27B + DS4F-0731 gridbook quants for DGX Spark — AQUA extends PrismaQuant sensitivity analysis to activation quantization (AURA was weight-only); Qwen3.8-27B-PrismaAQUA-5.5bit 22 GiB for Spark, tool-eval 93; m0l0 benchmarks on downclocked (2 GHz) single Spark: 2311 pp/s, 20.0 tg/s single, 37.6 @c2, 66.4 @c4 (with MTP); DS4F-0731-PrismaAQUA-gridbook-87GB for single Spark: 87 GB single-file checkpoint OOMs on 128 GB UMA (load peaks ~2× file size, CPU staging + UMA allocation share pool), re-sharding to 11×8GB fixes load; no-EOS degenerate repetition bug reported; vedrin.jeliazkov recipe with eugr vLLM 0.27.2rc1, gpu-mem-util 0.92, max-model-len 98304, fp8 KV, FULL_AND_PIECEWISE cudagraph, instanttensor load, 100+ tok/s concurrent (thinking=false); erlendboe 19-20 tok/s single-stream thinking-on (correct eugr image), 36.9 tok/s @ 3 concurrent; tenari expects 19-20 tps decode, 750 tps prefill; MTP pending (tenari/RobTand, m0l0, vedrin.jeliazkov, erlendboe, wga472) | https://forums.developer.nvidia.com/t/380364 | 2026-08-17 |
| S-forum-nemotron35-lightning-arena | forum | NVIDIA Nemotron-3.5-Lightning-30B-A3B-NVFP4 Spark Arena Benchmark link — automated benchmark post claiming 116.84 tok/s text generation on DGX Spark with vLLM; single-post bot-generated benchmark page link, no recipe details or community discussion (kuscsik) | https://forums.developer.nvidia.com/t/379875 | 2026-08-12 |
