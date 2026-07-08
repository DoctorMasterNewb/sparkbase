# Nemotron-3

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-nemotron-rpc, S-swapper
> **updated:** 2026-07-08

NVIDIA Nemotron-3 — **hybrid Mamba-2 + attention MoE** (`nemotron_h_moe`). Most layers are SSM with a
few attention layers (2 KV heads), so KV is cheap and native context is huge. Two paths on GB10.

## Nemotron-3-Super-120B-A12B (Q8 GGUF) — llama.cpp RPC, 2-node

- `timteh673/Nemotron-3-Super-120B-A12B-Uncensored-GGUF` Q8_0 (**128 GB single blob** > 121 GB/node →
  must split). Engine: **croll83 `llama.cpp-dgx` fork** (sm_121a, registers `nemotron_h_moe`).
- **[proven]** 2-node **pipeline RPC** (`[[wiki/llama-cpp-rpc.md]]`): ~61 GB/node, **~10.5 tok/s**
  decode, coherent. Native ctx **1,048,576** (no YaRN) — ran 1M × 4 slots, ~43 GB free/node (KV cheap:
  Mamba-2 hybrid).
- **[proven]** **`--no-mmap` REQUIRED** (else unified-mem OOM-kill mid-load). RPC has no auth — fabric
  IP only. If decode garbles with the hybrid SSM, bias `--tensor-split` so all SSM layers stay on one
  node.
- **[proven]** **Gate on arch BEFORE the 128 GB download:** read `general.architecture` via a 2 MB
  HTTP range read.

## Nemotron-3-Nano-Omni (vision/omni) — the worker vision/omni unit

- **[proven]** `AEON-7/Nemotron-3-Nano-Omni-AEON-Ultimate-Uncensored-BF16` — 30B NemotronH hybrid
  (Mamba2 + attn + 128-expert MoE), BF16, omni (text + image RADIO + audio Parakeet), ctx 200k. Image
  `ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1`. Single-node on the worker. (Also NVFP4
  variants exist via Atlas: `nemotron-3-nano-nvfp4`, `nemotron-3-super-nvfp4`.)

## See also
`[[wiki/llama-cpp-rpc.md]]` · `[[wiki/attention-and-kv-cache.md]]` (hybrid SSM = cheap KV) · `[[wiki/engines.md]]`
