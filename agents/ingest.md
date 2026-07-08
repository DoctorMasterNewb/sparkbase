# Ingest: turning a source into knowledge

How to fold a new **DGX Spark forum post** or **linked GitHub repo** (or a report) into the KB with
honest provenance. Both stacks use this; the only difference is the tier you're allowed to reach
(hardware agents can verify and go higher — see [`stack-hardware.md`](stack-hardware.md); analysis
agents cap at `[reported]` — see [`stack-analysis.md`](stack-analysis.md)). Read [`../SCHEMA.md`](../SCHEMA.md)
for the ladder and page format.

## Procedure

1. **Register the source.** Add a row to `sources/README.md`:
   `| S-<slug> | <type> | <one-line description> | <URL or first-party: what/when> | <date> |`
   Pick the `type` honestly — it caps the tier this source can justify:
   - `forum` — a forum/Reddit/Discord post → `[conjecture]` (→ `[reported]` if it corroborates others)
   - `repo` — a GitHub repo/README/recipe, unverified → `[conjecture]` (→ `[reported]` if corroborated)
   - `report` — vendor spec, blog, research writeup → `[conjecture]`/`[reported]`
   - `first-party` — you ran it on a real Spark → `[reproduced]`/`[proven]` (hardware agents only)

2. **Extract only durable, GB10-relevant findings.** Skip anything generic (true of any GPU/vLLM) or
   ephemeral (a one-off user error). Keep what a future bring-up would want: hard constraints, exact
   flags/env/error-strings, model ids, image tags, numbers-with-their-config.

3. **Classify each finding on the ladder** (SCHEMA → evidence ladder). Default questions:
   - Is this asserted by just this one source? → `[conjecture]`.
   - Do other *independent* sources already say the same on the target page? → merge and raise to `[reported]`.
   - Did someone run it on a Spark (this source, or you)? → `[reproduced]`/`[proven]` (hardware only).
   - Does it contradict an existing claim? → don't overwrite; see step 5.

4. **Place it — merge, don't append.** Find the page via `index.md`. Fold the finding into the claim it
   belongs to (strengthen/tighten an existing line before adding a new bullet). Tag it. If it's genuinely
   new scope, create a page and add it to `index.md`. **Sanitize** as you write (no private
   hostnames/IPs/service-names/paths — SCHEMA → Sanitization).

5. **Handle conflicts explicitly.** If the new finding disagrees with an existing claim:
   - New source is *weaker or equal* tier → note the disagreement inline, keep both, flag for verification.
   - New source is a first-party `[proven]` result overruling a `[reported]` one → the old claim becomes
     `[superseded]` (with a successor pointer); the proven result takes the claim.

6. **Update headers + ledgers.** Add the `S-` id to the page's `sources:`, update `evidence:` if the
   dominant tier changed, bump `updated:`. Append one `log.md` entry: date, source id(s), pages touched,
   one line of what changed.

## Worked micro-example

> A forum thread says "MXFP4 GGUF dispatches to a native FP4 path on GB10, ~40 tok/s."

- Register: `| S-forum-mxfp4 | forum | claim: MXFP4 GGUF native FP4 path, ~40 tok/s | <url> | 2026-07-08 |`
- The `[proven]` claim on `quantization-on-gb10.md` is that **GB10 has no native FP4 compute** (first-party).
  The forum claim contradicts it and is weaker (forum vs first-party).
- Result: add a `[conjecture]` line under the quant page noting the forum claim **and** that it conflicts
  with the proven no-native-FP4 finding; queue "measure MXFP4 GGUF decode path on real HW" in `roadmap.md`.
  Do **not** promote or let it overwrite the proven claim.
