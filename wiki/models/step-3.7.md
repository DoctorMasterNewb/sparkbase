# Step-3.7-Flash-NVFP4

> **area:** model
> **status:** evolving
> **evidence:** proven
> **sources:** S-m3-20tps, S-swapper
> **updated:** 2026-07-08

`stepfun-ai/Step-3.7-Flash-NVFP4` — TP=2 reasoning model with MTP. Retired from production serving
(unused in practice); the 25.8 GB `vllm/vllm-openai:stepfun37` image was deleted on both nodes. Kept
here for the **MTP + cudagraph finding**, which is reusable.

## The reusable finding

- **[proven]** Step-3.7 is the model that motivated the "20 tok/s" target via **MTP speculative
  decode** — the path to ~32 tok/s. But MTP's gain depends on cudagraphs: **MTP gives ~0 gain
  WITHOUT cudagraphs**, and cross-node MoE on GB10 is cudagraph-walled
  (`[[wiki/cudagraphs-and-compile.md]]`). So MTP + cross-node on Spark doesn't deliver the speedup it
  does on hardware where capture works.

## Bring-up notes (archived)

- Image `vllm/vllm-openai:stepfun37` (Step-3.7 NVFP4 + MTP kernels).
- **[proven]** Cross-node **full CUDA-graph capture of the TP NCCL collectives hangs** → ran the
  worker rank with a no-cudagraph (`-nocg`) launch. **[reported]** NVIDIA forum 373163 reports
  ~32 tok/s @ 262k ctx for this model on 2× Spark with MTP — not reproduced here once the cudagraph
  wall is accounted for. eugr guidance: Step-3.7 wants `--no-ray`.

## See also
`[[wiki/cudagraphs-and-compile.md]]` · `[[wiki/multinode-tp-and-networking.md]]`
