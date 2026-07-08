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

## Adding a source

Append a row with a new `S-` id and its `type`, then ingest per [`../SCHEMA.md`](../SCHEMA.md) and
[`../agents/ingest.md`](../agents/ingest.md). Forum/repo/report sources cite a URL; first-party
sources cite the experiment (what/config/when), never a private path.
