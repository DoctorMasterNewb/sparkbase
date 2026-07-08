# Change log

Append-only. One entry per ingest/lint: date, source(s), pages touched, one line of what changed.

## 2026-07-08 — Public seed: sanitize + evidence-tag the initial KB
- Established sparkbase from a private GB10/DGX-Spark knowledge base: added the evidence ladder
  (`conjecture → reported → reproduced → proven → superseded`), the two-stack agent model
  (hardware vs analysis), SCHEMA.md, AGENTS.md + `agents/`, README + CONTRIBUTING.
- Ported all wiki pages: sanitized private setup (hostnames/IPs/service names/personal paths → role
  wording + examples) and tagged every claim on the ladder. First-party bring-ups → `[proven]`;
  external report/forum claims → `[reported]`/`[conjecture]`.
- Rebuilt `sources/README.md` with source types (forum/repo/report/first-party); S-ids kept stable.
