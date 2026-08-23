# Benchmark methodology — traps that produce confident wrong numbers

> **area:** benchmarks
> **status:** stable
> **evidence:** proven
> **sources:** S-b12x-ab, S-gb10-profile
> **updated:** 2026-08-23

Not GB10 knowledge — **harness** knowledge, kept here because a benchmark number is only as good as
the harness that produced it, and sparkbase's whole premise is that claims wear their evidence.
Every trap below was hit on 2026-08-22 during a first-party NVFP4 MoE backend A/B, and every one
produced a **plausible, well-formed results table** instead of an error.

**[proven] The meta-rule: a benchmark harness fails silently and reports success.** Exit code was 0
on every failure listed here. Verify that a measurement *happened* before interpreting what it says.
(S-b12x-ab)

## [proven] Trap 1 — sparkrun replays cached results across A/B arms

**Symptom.** Two arms return **byte-identical** numbers to one decimal in every cell. Reads as "the
flag changed nothing" — the most dangerous possible false result, because it is publishable.

**Root cause.** `sparkrun benchmark run` derives its `Benchmark ID` from **model + profile + host**.
It does **not** include the recipe file, the recipe `name:`, or any `-o` override. Arms collide on
one state directory, the second finds it complete, and re-exports the first's results without
benchmarking. The engine *does* boot with the requested flag — the serve log proves the kernel
changed — only the measurement is fake.

**What does NOT fix it** (all tried, all still replayed): `--fresh`; a separate recipe file per arm
with a distinct `name:`; `mv`-ing the state dir aside (works once, then silently fails because `mv`
cannot move `bench_X` into an archive already containing `bench_X` — and then replays the *previous*
arm's numbers under the next arm's name).

**Fix.** `rm -rf ~/.cache/sparkrun/benchmarks/bench_*` before **every** run, plus an assertion that
consecutive result files differ (md5). Reference: `run-repeats2.sh` in S-b12x-ab.

**Tell.** Genuine reruns differ by tenths. **Identical-to-the-decimal is proof of replay, not of
determinism.** Observed 4 times across 3 mitigations. (S-b12x-ab)

## [proven] Trap 2 — profile shape moves absolute throughput ~50%

Same box, same backend, same weights: `pp:[2048,8192]` gives **155.9** tok/s at c64; `pp:[2048]`
gives **235.5**. Interleaving 8192-token prefills spends the budget on prefill and depresses decode
~34%. **Never compare across profiles, and quote the profile with every tok/s figure.** A
"saturates by c16, compute-bound" reading of that model was partly this artifact. (S-b12x-ab)

## [proven] Trap 3 — alias-only `--served-model-name` ⇒ silent 404

llama-benchy addresses the endpoint by **HF repo id**. Serving only an alias yields
`Warmup failed: HTTP 404: The model 'org/Model' does not exist`; every task fails in ~2.7 s while the
server is healthy. Serve both: `--served-model-name {model} my-alias`. (S-b12x-ab)

## [proven] Trap 4 — `--exit-on-first-fail` (CLI default) overrides the profile

`exit_on_first_fail: false` in the profile is ignored; the CLI default aborts the arm on the first
bad cell. Pass `--no-exit-on-first-fail`. (S-b12x-ab)

## [proven] Trap 5 — never edit a bash script while it runs

Bash reads scripts **by byte offset as it executes**; patching a running script shifts bytes under
the live interpreter, which resumes mid-token (`unexpected EOF while looking for matching '"'`) —
*after* the arm that already passed, so it looks like partial success, and `bash -n` on the file
afterwards passes. Run an immutable copy. (S-b12x-ab)

## [proven] Trap 6 — vLLM's real log is not `docker logs`

Under sparkrun, `docker logs` carries only the NGC banner; the engine log lives at
**`/tmp/sparkrun_serve.log` inside the container** and dies with it (`--rm`). Capture it live with
`docker exec`. **Always record which kernel actually served** — otherwise an arm that silently fell
back is indistinguishable from one that switched, and you publish the fallback's numbers under the
new kernel's name. (S-b12x-ab)

## [proven] Trap 7 — prove the GPU was healthy, don't assume it

Sample clocks/power for the whole run and publish the range. A healthy GB10 under load shows a
**varying** clock and real watts (measured: 2346–2515 MHz, 17.5–96.3 W across 158 under-load
samples). The power-controller wedge is a clock **pinned to one exact value at ~12–14 W with no
throttle flag** (`[[wiki/platform-gb10.md]]`). **[reported]** On a node pending RMA, discard results
on any failure — especially a sudden reboot, which is that fault's usual presentation. (S-b12x-ab)

## [proven] Trap 8 — knobs that are advertised, accepted, and silently inert

| knob | looks like | actually |
|---|---|---|
| `VLLM_NVFP4_GEMM_BACKEND` | documented env var | **does not exist** on some builds; silent no-op |
| `--linear-backend flashinfer_b12x` | in `--help`, accepted into `KernelConfig` | **no kernel exists** for NVFP4; engine dies at startup |
| `VLLM_TORCH_PROFILER_DIR` | upstream profiler switch | **inert** where routes are gated on `--profiler-config`; `/start_profile` returns **404** |

**Verify the knob took effect, not that you passed it** — read the engine's config echo
(`non-default args:` / `KernelConfig(...)`) or the HTTP status. A run whose central variable was
inert yields a clean table that means nothing. (S-gb10-profile)

## [proven] Trap 9 — the GB10 profiling recipe (the standard one does not work)

- **`ncu` needs `--cap-add=SYS_ADMIN`** on the container.
- **No memory-controller counters exist on sm_121**: `dram__*` = **0 metrics**, `fbpa__*` = **0**.
  Every roofline recipe using `gpu__dram_throughput` / `dram__bytes.sum` returns **`n/a`**.
  **Substitute `lts__t_sectors_lookup_miss x 32 B` / duration** for achieved bandwidth
  (`lts__` 2031, `l1tex__` 461, `sm__` 367 metrics are present).
- **torch profiler**: `--profiler-config '{"profiler":"torch","torch_profiler_dir":"…","torch_profiler_with_stack":false}'`,
  then POST `/start_profile` … `/stop_profile`. Traces land **inside the container** — `docker cp`
  them out before teardown (`--rm`) or they are lost.
- **Demangle before bucketing.** Our largest entry was `std::enable_if<!(false)...` at 69% — a
  truncated cuBLAS **GEMV** template. Prefix-bucketing would have hidden the whole finding.
  (S-gb10-profile)

## [proven] Trap 10 — measure adoption, not just correctness or latency

**Report the fraction of device time spent in the kernel you believe you are testing.** From the
Atrex kernel-agent paper (arXiv 2607.14541), whose DSL-adoption metric exposed 84.8% correctness at
43.8% adoption. On GB10 it catches two failures that correctness and latency both pass: an engine
that **fell back** to another backend, and a **checkpoint that is barely quantised** — a "NVFP4
model" measured **19% NVFP4 adoption** (`[[wiki/quantization-on-gb10.md]]`). (S-gb10-profile)

## The discipline

1. Assert the measurement happened (distinct results).
2. Record which kernel served (engine log, per arm).
3. Record hardware state (clock/power range).
4. Quote the profile with every number.
5. Establish the noise floor **before** believing a delta — see `[[wiki/benchmarks.md]]`.

## See also
`[[wiki/benchmarks.md]]` · `[[wiki/quantization-on-gb10.md]]` · `[[wiki/platform-gb10.md]]`
