# llama.cpp on GB10 (incl. 2-node RPC)

> **area:** llama.cpp
> **status:** stable
> **evidence:** proven
> **sources:** S-nemotron-rpc
> **updated:** 2026-07-08

llama.cpp is the path for **GGUF** checkpoints and for archs vLLM/Atlas don't support (e.g.
`nemotron_h_moe` hybrid Mamba-2+attn MoE). Its 2-node story is **pipeline RPC**, not tensor-parallel —
lower throughput than vLLM TP, but it splits a model too big for one node with minimal fuss.

## Build (sm_121a, with RPC)

```bash
cmake -B build-rpc -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real \
      -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_RPC=ON
cmake --build build-rpc --target llama-server rpc-server -j
```
- **[proven]** Default builds ship `GGML_RPC=OFF` — you must rebuild for the `rpc-server` target.
- **[proven]** Keep `-DGGML_CUDA_FA_ALL_QUANTS=ON` for quantized KV (`-ctk/-ctv q8_0`).
- **[proven]** Binaries are identical-arch for both GB10s → **scp `rpc-server` (+ matched
  `libnccl.so.2`, it links NCCL) to the worker** instead of rebuilding. Use the croll83
  `llama.cpp-dgx` fork (has sm_121a + nemotron arch) or upstream.

## 2-node RPC serving

- **[proven]** **Worker:** `rpc-server -H <worker-ip> -p 50052 -c` (example: `-H 192.168.0.9` —
  substitute your fabric IP). The `-c` caches received shards locally (`~/.cache/llama.cpp/rpc/`), so
  only the **first** launch pays the one-time weight push over the fabric (~64 GB ≈ 30–60 s);
  thereafter it's warm. **The full GGUF lives only on the head** — RPC pushes each backend's tensors
  over the wire; the worker never reads the file.
- **[proven]** **Head:** `llama-server -m <model>.gguf --rpc <worker-ip>:50052 -ngl 99 --tensor-split
  0.5,0.5 --no-mmap --host 0.0.0.0 --port 8888 -ctk q8_0 -ctv q8_0 -fa on --jinja` (example
  `--rpc 192.168.0.9:50052` — substitute yours). Device order is `[CUDA0(head), RPC0(worker)]`;
  `--tensor-split` ≈ GB/node.
- **[proven]** **RPC has no auth** — bind only to the private fabric IP, never the management LAN.

## Hard rules

- **[proven]** **`--no-mmap` is REQUIRED.** With mmap the head's ~61 GB non-reclaimable CUDA buffer
  **plus** the mmap page-cache of the (e.g.) 128 GB GGUF fills all 121 GB unified → silent OOM-kill
  mid-load (the worker never even receives a tensor). `--no-mmap` reads straight into buffers, no
  lingering cache. This is a direct consequence of unified memory (`[[wiki/platform-gb10.md]]`).
- **[proven]** **Size `--tensor-split` for head asymmetry.** The head also runs the OS + server
  process; if it OOMs, shift toward the worker (`0.45,0.55`). Unified-mem OOM = **hard reboot**, so
  size conservatively.
- **[proven]** **Gate bring-up on arch support BEFORE downloading.** Read `general.architecture` from
  the GGUF header via a ~2 MB HTTP range request; confirm the fork/upstream registers that arch
  (`nemotron_h_moe` → croll83 fork registers it). A 128 GB download you can't serve is expensive.

## Hybrid-SSM watch-item

- **[proven]** llama.cpp's recurrent/SSM state cache is per-sequence and co-located with its layers, so
  splitting **SSM layers across the RPC boundary** is less battle-tested than splitting attention. If
  decode is garbled with a hybrid (Mamba-2) model, bias `--tensor-split` so **all** SSM/recurrent
  layers land on one node.

## Measured

- **[proven]** Nemotron-3-Super-120B-A12B Q8_0 (128 GB, hybrid SSM+MoE), 2-node RPC, ~61 GB/node, `-c`
  1M ctx × 4 slots: **~10.5 tok/s** single-stream decode, coherent, no NaN. Modest — 12B active over a
  cross-node RPC hop — but it runs a 128 GB model that fits on neither node alone.

## See also
`[[wiki/multinode-tp-and-networking.md]]` · `[[wiki/engines.md]]` · `[[wiki/models/nemotron-3.md]]`
