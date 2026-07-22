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

## Adding a source

Append a row with a new `S-` id and its `type`, then ingest per [`../SCHEMA.md`](../SCHEMA.md) and
[`../agents/ingest.md`](../agents/ingest.md). Forum/repo/report sources cite a URL; first-party
sources cite the experiment (what/config/when), never a private path.
