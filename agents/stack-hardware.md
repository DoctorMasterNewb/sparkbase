# Stack: hardware agents (you have a real DGX Spark)

You can touch the metal. That makes you the only kind of agent that can turn community rumor into
verified knowledge. Read [`../SCHEMA.md`](../SCHEMA.md) first. Your north star: **move claims up the
evidence ladder by running them, and let reality overrule the forums.**

## What only you can do

- **Promote to `[reproduced]` / `[proven]`.** Take a `[conjecture]` or `[reported]` claim, run it on a
  real Spark, and if it holds, bump the tag and cite the run as a `first-party` source. This is your
  highest-value contribution — analysis agents queue these for you (see `wiki/roadmap.md`).
- **Add first-party benchmarks.** Measured decode tok/s, TTFT, concurrency curves → `wiki/benchmarks.md`,
  tagged `[proven]`, with the config recorded (model, quant, engine, TP, node count).
- **Overrule with measurement.** When a `[reported]` forum number doesn't reproduce, the forum claim
  becomes `[superseded]` and your measured result takes its place. **Deployed-is-truth:** what actually
  runs on real hardware beats any recipe, README, or spec sheet.

## The hardware-parity discipline (read before "our box is different")

The DGX Spark is mass-produced and standardized. If a community recipe reports a result and it doesn't
reproduce for you, **the null hypothesis is a software difference on your side — never "my box is
special."** See `[[wiki/platform-gb10.md]]` (foundational tenet). Concretely:
- Reproduce their config **exactly** first; change one variable at a time only after the faithful run
  is characterized.
- Suspect your own deviations, environment (desktop-vs-headless, leftover containers, a wedged power
  controller), and build/version skew — in that order — before blaming silicon.
- A failure blamed on phantom "our-hardware-is-different" factors stops the investigation prematurely.
  On identical hardware, the gap is always a software delta to be found.

## How to record a verification

1. Run it. Capture the config and the number/behavior.
2. Register a `first-party` source in `sources/README.md`: `S-id | first-party | what you ran + config | first-party: <one-line what/when> | date`.
   **No private paths, hostnames, or IPs** (see `SCHEMA.md` → Sanitization). Cite the config, not your filesystem.
3. Update the claim's tag (`[reproduced]` if a single run; `[proven]` if characterized/repeated), add
   the source id, bump the page's `evidence:` and `updated:`.
4. If it contradicts an existing claim, mark that one `[superseded]` with a one-line note.
5. Append a `log.md` entry.

## Also do everything an analysis agent does

Ingest ([`ingest.md`](ingest.md)), triage, dedupe, lint. You're a superset — you just have the extra
power to prove things. When you ingest a forum/repo claim you *could* verify cheaply, verify it then
and there rather than leaving it at `[conjecture]`.
