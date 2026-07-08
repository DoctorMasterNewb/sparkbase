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

## Adding a source

Append a row with a new `S-` id and its `type`, then ingest per [`../SCHEMA.md`](../SCHEMA.md) and
[`../agents/ingest.md`](../agents/ingest.md). Forum/repo/report sources cite a URL; first-party
sources cite the experiment (what/config/when), never a private path.
