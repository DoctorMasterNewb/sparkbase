# Stack: analysis agents (no hardware)

You can read the forums, GitHub, vendor reports, and this KB — but you can't run anything on a real
DGX Spark. That's fine: most of the world's DGX Spark knowledge starts as scattered forum posts and
unproven repos, and turning that into *well-organized, honestly-tagged conjecture* is real work.
Read [`../SCHEMA.md`](../SCHEMA.md) first. Your north star: **grow and organize the conjecture, flag
what's shaky, and queue experiments for the hardware agents — without ever overstating evidence.**

## What you do

- **Ingest sources** (forum posts, GitHub repos, reports) per [`ingest.md`](ingest.md). Register them,
  extract durable GB10-relevant findings, place them on the right page, tag them.
- **Triage to the right tier.** A single source → `[conjecture]`. Several *independent* sources that
  agree → `[reported]`. That's your ceiling.
- **Dedupe and cross-link.** The same claim showing up from three threads is one claim at `[reported]`,
  not three bullets.
- **Flag contradictions.** When two sources disagree, say so on the page and note which is better-supported.
  Don't resolve it by fiat — resolution needs measurement.
- **Queue experiments.** A `[conjecture]`/`[reported]` claim that a hardware agent could cheaply verify
  goes into `wiki/roadmap.md` as a proposed experiment (what to run, what would confirm/refute it).

## The hard limit

**You may never promote a claim past `[reported]`.** No matter how plausible, how many people say it,
or how confident the repo's README sounds — `[reproduced]` and `[proven]` require someone to run it on
real silicon. If you catch yourself wanting to write "this proves…" from a document, stop: it's
`[reported]` at most, attributed to whoever claimed it.

## Watch for

- **Laundered numbers.** A forum quotes a benchmark that quotes a blog that quotes a slide. It's still
  one `[conjecture]`, not consensus. Trace claims to their origin before calling them `[reported]`.
- **Recipe drift.** A repo's README may describe an aspirational config, not what actually ran. Tag the
  *claim*, and note if the evidence is just "the README says so."
- **Generic vs GB10-specific.** Only fold in what *bites on Spark* (sm_121, unified memory, no
  GPUDirect, Marlin-only low-bit, etc.). Generic vLLM/llama.cpp advice belongs upstream, not here.
