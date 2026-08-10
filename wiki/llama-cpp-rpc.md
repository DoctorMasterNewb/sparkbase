# llama.cpp on GB10 (incl. 2-node RPC)

> **area:** llama.cpp
> **status:** stable
> **evidence:** proven
> **sources:** S-nemotron-rpc, S-forum-m3-llamacpp-2x, S-forum-dsv4-llamacpp-fan, S-forum-dsv4-0731-gguf
> **updated:** 2026-08-10

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
- **[conjecture]** MiniMax-M3 426B MoE UD-IQ4_XS GGUF (~194 GiB, ~97 GiB/node), 2-node RPC,
  `--split-mode layer`: **~10.7 tok/s** decode, ~590 tok/s prefill @ `--ubatch-size 2048`, 65K ctx
  (KV q8_0 ≈ 45 KB/token). Tool-calling via a **hybrid chat template** (M3 native body + M2 tool-call
  format; llama.cpp PR #24523's parser can't read M3's native format). First load ~13–25 min (cached
  after). Build: `CUDA_ARCH=121` from source, `aarch64`/GCC 13 (S-forum-m3-llamacpp-2x, karol.spark).
- **[conjecture]** DeepSeek-V4-Flash-0731 UD-IQ2_M GGUF on single HP ZGX (GB10), llama-server,
  `--n-gpu-layers 999 --flash-attn on --ctx-size 524288 --parallel 4 --cont-batching --batch-size 2048
  --ubatch-size 1024 --jinja --threads 10 --no-mmap`: **16.2 tok/s** tg32 (pp2048 390 tok/s, ttfr
  4860ms). Decode degrades with depth: 16.2→15.86 @ 4K, →15.82 @ 8K, →15.26 @ 16K. Firmware update
  improved thermals to 71°C / 75W with no shutdown (S-forum-dsv4-llamacpp-fan, chrm). The `--no-mmap`
  flag is consistent with the proven UMA requirement. The IQ2_M (2-bit UD quant) allows the ~440B
  model to fit in a single 121 GB node. Prefill is low (390 tok/s) — llama.cpp's CPU-side processing
  on Grace limits prefill vs vLLM's CUDA prefill path. See `[[wiki/benchmarks.md]]` → Batch 58.

- **[conjecture]** DeepSeek-V4-Flash-0731 UD-IQ2_M GGUF (Unsloth release) on single Spark via
  `llama-server` (S-forum-dsv4-0731-gguf, chriswalz86): `--n-gpu-layers 999 --flash-attn on
  --ctx-size 262144 --parallel 2 --batch-size 2048 --ubatch-size 512 --jinja --reasoning off
  --no-repack --cache-type-k f16 --cache-type-v f16 --temp 0.6 --top-p 0.95 --top-k 0 --min-p 0.0`.
  Works without issues. The `--no-repack` flag disables llama.cpp's default tensor repacking.
  Unsloth also published **UD-Q8_K_XL** (162 GB, "full precision lossless", 7 GB larger than Q4) —
  too large for single Spark, needs 2-node RPC. Consistent with the S-forum-dsv4-llamacpp-fan
  recipe (same quant, same flags). No tok/s reported in this thread.

## See also
`[[wiki/multinode-tp-and-networking.md]]` · `[[wiki/engines.md]]` · `[[wiki/models/nemotron-3.md]]` · `[[wiki/benchmarks.md]]`
