# Wiki schema (the configuration file of this LLM-Wiki)

> Merged wiki (2026-11-03): aggregated from `lean4_{stockpredict,foursquare,pcap,geolife,booking,snap}/wiki`.
> Pages that differ per build keep one section per build (flow → stock → travel → GeoLife → snap);
> `log.md` is the union of all six logs, de-duplicated and sorted by date.

This directory is an **LLM-Wiki** in the sense of Karpathy's note
(gist `442a6bf555914893e9891c11519de94f`): a persistent, compounding knowledge
base that an LLM agent builds and maintains, sitting between the humans and the
raw sources. The human curates sources and asks questions; **the LLM writes and
maintains every page here**. Knowledge is compiled once and kept current, not
re-derived per query (contrast with RAG).

## Three layers

1. **Raw sources** (immutable, never edited by the LLM). This wiki is the
   AGGREGATE of the six sibling builds under `algo_lean4/`; paths are relative
   to each build (`lean4_<name>/`):
   `lean4_snap/facebook_full_features.graphml` (SNAP graph),
   `lean4_geolife/geolife_data.tsv` (timestamps UTC),
   `lean4_booking/` (booking dataset; shares the GeoLife-era wiki),
   `lean4_foursquare/country_transfer.tsv` (+ `data_ingest/dataset_TIST2015/`, its origin corpus),
   `lean4_stockpredict/stock_normalized_train.tsv`,
   `lean4_pcap/flow.jsonl` (CIC-IDS-style Joy flows) and the original `nitroba.pcap`,
   each build's `goal.md`, `METHOD.md`, dbt marts in `artifacts/enclave.duckdb`,
   model artifacts in `models_out/`, Lean results in `lean/out/`.
2. **The wiki** — this directory. Markdown pages owned entirely by the LLM.
3. **The schema** — this file. Conventions and workflows; co-evolved with use.

## Conventions

- One page per entity (host, dataset) or concept (motif, mechanism).
  File names: short-kebab-case, `.md`.
- Every page starts with YAML frontmatter:
  `title`, `type` (entity | concept | component | synthesis), `updated`
  (YYYY-MM-DD), `sources` (list of raw-source paths or page links).
- Pages link to each other with `[[wikilinks]]` (Obsidian-compatible;
  the target is the file name without `.md`).
- `index.md` is the catalog: every page, one line + link, grouped by category.
  **Update it on every ingest.** Read it first when answering a query.
- `log.md` is append-only. Entry prefix: `## [YYYY-MM-DD] <op> | <title>`
  where `<op>` ∈ ingest | query | lint | update — plus the operations the log
  has grown to use: fix (a defect found and corrected), release (a DP release or
  export), build (a pipeline instantiation), models (a minimodel retrain),
  measure (a re-measurement that changes a documented claim), extend (a new
  capability), and `a+b` for an entry that is both (e.g. `release+synth`). Parseable:
  `grep "^## \[" log.md | tail -5`.

## Operations

- **Ingest**: read a new source (an EDA run, a build result, a new mart),
  write/update the relevant pages, update `index.md`, append to `log.md`.
  A single source may touch many pages — that is the point.
- **Query**: read `index.md`, drill into pages, synthesize with citations.
  Good answers get filed back as new pages.
- **Lint**: look for contradictions, stale claims (e.g. a page citing a model
  version older than `models_out/latest.txt`), orphan pages, missing
  cross-references; fix and log.
