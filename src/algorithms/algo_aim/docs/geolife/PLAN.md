# GeoLife plan: DP outings artifact, teleport, and method M

Current plan and next steps. Rough history and decisions: `PROCESS.md`. Interfaces between the pieces: `INTERFACES.md`.
Superseded plans (home/work version, the outings draft, the first task board): `archive/`. Background reading:
`../../research/inspectable_synthetic_data_artifact_proposal.docx`, `p0-feasibility.md` (Overture and routing).

Reading order: section 1 (goal), section 8 (where we are), section 9 (next steps). Sections 2-7 are the design.

## 1. Goal and success criterion

Real GeoLife traces exist only inside an air-gapped network. Only a small, inspectable, differentially private **artifact**
may leave it. Outside, we generate **fully synthetic** users and GPS traces in a city of our choosing from that artifact alone, and
develop an unsupervised analysis **M** on the synthetic data. M finds, from raw traces and a public map:
(a) places of interest per user, (b) common places and common road segments (shared by several users), (c) relations between
users (who shares places and routes).

**Success:** M, developed and tuned only on synthetic data, is run unchanged on the real data and behaves the same - the same kinds
of outputs with the same distributions and error profile. M uses map-relative features only (place categories, road segments,
dwell, hours), never absolute coordinates.

Not goals: one-to-one correspondence between synthetic and real users (that would be a per-user transform, not synthesis); hiding
which city the data came from by DP (teleporting is obfuscation of provenance, not a privacy mechanism).

## 2. The data and the model of it

GeoLife is mostly **outings**, not daily routines: 182 users, 18.7k real outings (after filters), no stable overnight home or workday.
So the model has no home/work roles (the first version did and failed on real data, see `PROCESS.md`).

- A **user** is a set of outings, observed for weeks to years.
- An **outing** is one trajectory (a gap above 2 h starts a new one; at least 5 min and 500 m). It has a mode, a length, and a few **visits**:
  stops (10 min or more within 200 m) plus the start and end.
- A visit is snapped to an Overture place within 150 m, else to an H3 cell. Time uses local time (UTC+8; GeoLife timestamps are GMT).
- Real regime (EDA): 1.38 stops per outing, revisit rate 59% at H3 res 8, outing starts peak at 08h and 17-18h, stops skew to weekends,
  median sampling interval 2 s, a heavy tail of users and a few hub places (6 places with 51-67 users each).

## 3. Setting: three stages, four zones (as `snap/`)

| Stage | Zone | Does | Script |
|---|---|---|---|
| 1 | inside | real traces -> artifact | `geolife/inside/fit_artifacts.py` |
| 2 | outside | artifact (+ public map of any city) -> synthetic traces, outings and planted truth | `geolife/outside/generate_synthetic.py` |
| 3 | eval | synthetic + real -> M comparison and metrics | `geolife/eval/score.py` |

`transfer/` holds the artifacts and the shared code (types, artifact I/O, tokenization, road graph and matcher). `eval/` holds M.
Import rules and the boundary test: `CLAUDE.md`, `tests/geolife/test_boundary.py`. Layout: `geolife/README.md`, `src/geolife/README.md`.

## 4. Privacy model

- **Unit:** one user (all their outings). Adjacency: add/remove one user. Contributions are capped per user (100 outings, 50 stop cells
  at res 8, 1000 traversed cells at res 8), so sensitivities are exact.
- **Mechanism:** Gaussian noise on clipped per-user count tables, zCDP accounting, epsilon 6 at delta 1e-6 by default; a ledger lists every table.
  Everything after the measurement (fit, generation, teleport, render) is post-processing.
- **Public vocabulary:** the keys (H3 cells, categories, buckets) come from the public map, so vocabulary selection leaks nothing.
  Fine levels are released only under parents whose noisy count clears a threshold.
- **Outside the guarantee** (listed as `caveats` in `privacy.json`): every public setting chosen by looking at the real data (gap, stop rule,
  caps, bins, hierarchy, snap radius, which tables are kept, the 3-bin time design, the sensor bin edges and caps). The renderer now draws
  from the released DP sensor histograms; the hand-set fallback values for noise-only tables are listed as a caveat (step 1 done, D20).
- Several artifact seeds are noise replicates. Releasing more than one lets anyone average them; for a real release use one seed.
  Policy implemented (step 7, D22): `fit_artifacts.py --release` emits exactly one seed and stamps the policy into `privacy.json`;
  the measured cost of the 3 committed dev seeds (noise std / sqrt(3); honest accounting eps 11.06) is in `geolife/eval/reports/privacy-attacks.md`.
- Empirical linkage and membership attacks on the generated data exist (`src/geolife/eval/attacks.py`, step 7, D21; results in
  `geolife/eval/reports/privacy-attacks.md`). They are a sanity check, not a substitute for the guarantee.
- Artifacts can be signed (optional HMAC-SHA256 over the manifest, `artifact_cli sign/verify --key`; shared-key authenticity,
  not a public-key signature; the key stays inside and reaches the outside verifier over the trusted code channel). Step 7, D23.

## 5. Pipeline design

**Inside (stage 1).** Load and clean (invalid points, speed spikes) -> pseudonymise -> outings and visits -> caps -> noisy tables -> ledger -> artifact.
Tables (26 since D27): distinct users per place (H3 res 6/7/8) and per traversed cell (res 6/7), hierarchical release; outing shape (stops, duration, length, mode,
detour, dwell, hop); per-user histograms (outings, distinct places, active days, revisit rate); outing starts
on coarse time bins (3 time-of-day bins x weekday/weekend since step 2, D19); sensor tables measured from the raw points (sampling
interval, gaps, speeds by mode, heading, stop jitter, points per outing; step 1, D20: 20% of the budget). Weighted budget split from public quantities with an equal-split floor. Noise-only tables are not released
(v2b dropped place_time_users / stop_transitions_r6 / trav_users_r8-r9; D27 dropped cat_time_visits, cat_transitions and sensor_step — category mix and profiles are generator defaults now).

**Outside (stage 2).** Fit a small `params` dict from the noisy tables (defaults where a table is mostly noise) -> sample users with
tastes, activity regions, latent groups with favourite places, revisits, co-travel pairs, and bursty observation windows -> **teleport** latent
places onto the target city's places (Hungarian assignment per category using popularity rank and density, same mapping for every user) ->
route on the target road graph -> render GPS (sampling, noise, gaps, stop jitter) -> write traces and the planted truth.

**Eval (stage 3).** M v2: stops clustered and merged on the public map -> places of interest (with an 8-bin visit-time profile) -> common places
and segments (k or more distinct users; HMM map matching where a road graph exists, else res-9 cells) -> co-visit graph (rarity-weighted,
significance against a popularity-only null, FDR) -> Louvain communities. Scored against the planted truth on synthetic data.
On real data only distribution-level statistics and split-half reliability are available (no labels).

## 6. Artifact layout

```
artifact/
  manifest.json        format version, learner, SHA-256 of every file
  privacy.json         adjacency, epsilon, delta, rho, ledger entry per table, caveats
  schema.json  preprocessing.json  grounding.json  generation.json  diagnostics.json
  model/behaviour/<table>.parquet + <table>.meta.json     key, noisy value; sigma, sensitivity, rho
  model/behaviour_summaries.json
  model/embeddings/    (empty: embedding tier is future work)
```
JSON and Parquet only, no pickle; validators reject non-allowlisted keys and raw coordinates. Current artifacts: `geolife/transfer/artifacts/outings_v2_seed0-2`.

## 7. Evaluation

1. **Planted-truth recovery on synthetic data** (precision/recall of places of interest, common places, common segments, relation edges, community ARI),
   with dev and held-out draws, and sweeps over GPS noise, gaps, observation window, number of users.
2. **Real vs synthetic distributions** (the transfer test): places and outings per user, stops per outing, outing length and duration, revisit rate,
   users per place, co-visit degree, users per segment; compared with total variation. Acceptance thresholds are not set yet (open decision).
3. **Split-half reliability** of M on real data and on synthetic data (does M reproduce itself across halves of a user's outings). Real baseline:
   co-visit edges Jaccard 0.38 (11x chance), places of interest 0.43 random / 0.30 time halves.
4. **Visual and statistical comparison of real vs synthetic** (new `geolife/eval/notebooks/viz_comparison.ipynb`, modelled on SNAP's, with library code in
   `src/geolife/eval/`). Where both are in the same frame the comparison is direct. To get that, also generate into the source city (Beijing) from the same artifact, so
   real and synthetic share lon/lat. Teleported cities are compared in centred metric frames (metres from each dataset's centroid, same scale). Panels:
   - *Space:* lon/lat density maps side by side and as a difference map; marginal lon and lat histograms with KS; a few example trajectories of real vs synthetic users.
   - *Moments and shape:* mean, standard deviation, skewness, kurtosis and quantiles (p1, p5, p25, p50, p75, p95, p99) tabulated for point-level quantities (sampling interval, step length,
     speed overall and by mode, heading change, positional jitter at stops, gap length) and outing-level quantities (length, duration, stops per outing, dwell, points per user, outings per user,
     active days); ECDF overlays and QQ plots for each.
   - *Time:* hour-of-day and weekday/weekend profiles of outing starts and stops; time-bin shares.
   - *Social/place structure:* users per place, users per segment, revisit rate, co-visit degree distribution (real vs synthetic), and the same for split halves.
   - A summary table of distances (KS, Wasserstein, total variation, relative error of moments) so progress can be tracked run to run.
5. **Privacy:** ledger review and empirical attacks.

## 8. Where we are (2026-11-02)

| Piece | State |
|---|---|
| DP core, artifact I/O, leakage and provenance guard, inspector | done, tested |
| Outing tokenization, EDA of all 182 users, public parameters | done |
| Learner v2 (about 20 behaviour tables + 9 sensor tables, weighted budget 80/20, 3 time bins, ledger) and 3 committed artifacts | done (steps 1 and 2, D19/D20); time tables now on 3 bins x daytype, SNR per table in `PROCESS.md` 2026-11-02 (steps 1-2) |
| Generator v2 (groups, revisits, co-travel, teleport, routed GPS render) | works on toy city, Amsterdam and Beijing; revisit rate, outing length and users-per-place are calibrated against the fitted artifact (steps 4-5, D24-D26); planted-stats check passes on the learned artifacts |
| Method M v2 | works on synthetic (places of interest P/R 0.88/0.84 dev, 0.74/0.72 held-out; segments 0.96/0.97; edge AUC 0.96/0.87); run on real with the cell proxy only |
| Zone restructure, boundary test, stage scripts | done |
| Real-vs-synthetic comparison (step 3: `eval/compare.py` + `viz.py`, notebook, `score.py --viz`, `--city beijing` generation) | done; "before" baseline in `PROCESS.md` ("Real-vs-synthetic baseline (before sensor tables)") |
| GPS rendering | **done (steps 1, 5)**: sampling period, per-user gap rate, gap lengths, stop jitter, speeds by mode drawn from the released DP sensor histograms (inverse CDF); receiver noise is AR(1) with in-stop drift (D26); hand-set defaults only as per-table fallback |
| Real-vs-synthetic transfer | **first pass done (steps 4-5, D24-D26)**: outing length KS 0.37 -> 0.09, heading-change median 121 -> 5.0 deg (real 3.5), gap lengths and stop jitter KS ~0.12, hour-profile TV ~0.1-0.2, co-visit degree 18.1 vs 19.2. Remaining: points per user ~2.7x, distinct places per user ~2.1x, dwell tail, pooled speed mean (`PROCESS.md` steps 4-5 table) |
| Privacy hardening (step 7) | **done 2026-11-02** (D21-D23, D27): empirical attacks (`src/geolife/eval/attacks.py`, report `geolife/eval/reports/privacy-attacks.md`), ledger review (`geolife/eval/reports/ledger-review.md`, all numbers OK, caveat gaps fixed forward), HMAC artifact signing, one-seed release policy, zero-signal budget dropped (D27: cat_time_visits, cat_transitions, sensor_step; artifacts regenerated, 26 tables, caveat fixes included) |
| Housekeeping (step 6) | **done 2026-11-02** (see `PROCESS.md` step-6/D27 section): v1 code retired (survivor map in `PROCESS.md`), outings export carries `cells`/`segments`, clean-room test tz allowlist; full-env check run with the system python (no `uv` on the machine; synthcity env still untested) |

## 9. Next steps (known problems first; M is last)

Priorities set by the user on 2026-11-02: first the GPS rendering from real noise distributions, then 3 time bins, then richer visual comparison, then the rest. **M comes last.**

**1. Render GPS according to the real noise distributions (most important). DONE 2026-11-02 (see `PROCESS.md` steps 1-2 section; D20).**
   Today the renderer uses hand-set defaults (noise about 6 m, stop jitter about 25 m, about 5 gaps per hour, speeds by mode), copied from the EDA. Replace them by distributions measured from the real raw points
   and released through the artifact:
   - *Inside:* a new sensor-measurement stage on the raw points (they are not in `UserOutings`, so it needs its own reader next to `inside/build_outings.py`). Per-point statistics, each with a bounded per-user
     contribution (cap the number of points or steps sampled per user with a deterministic rule; histograms with fixed public bin edges): sampling interval (log bins), gap length and gap rate per hour, step length and
     speed by mode (labelled and heuristic modes), heading change, positional jitter at stops (spread of points inside a stop around its centre), points per outing. Noise and accounting as for the other tables (zCDP,
     one ledger entry per table, budget shares from public quantities); decide the split between the behaviour tables and the sensor tables.
   - *Transfer:* new table names and keys in `schema_v2.py`, allowlisted `generation` keys, a ledger caveat for the capped sampling and the bin edges.
   - *Outside:* `render_v2` draws sampling intervals, gaps, jitter, noise and speeds from the released histograms (inverse-CDF sampling), falling back to the old defaults for any table that is mostly noise; draw
     them per user (some users sample at 1 s, some at 5 s) if the per-user histogram shows that heterogeneity.
   - *Check:* the visual comparison of step 3 (sampling interval, speed, jitter and gap panels) must show the synthetic distributions close to the real ones. Until the tables exist, list the hand-set values as a
     caveat in `privacy.json`.

**2. Three time bins instead of two. DONE 2026-11-02 (D19 edges; see `PROCESS.md` steps 1-2 section).**
   Released time tables move from am/pm x weekday/weekend to **3 time-of-day bins x weekday/weekend**. Proposed edges, to be fixed from public reasoning about the day before looking at any new SNR (the EDA shows
   outing starts peaking at 08h and 17-18h local): morning [05,11), midday [11,16), evening/night [16,05). Add `TOD_3` and a local-bin helper to `transfer/types.py` (keeping `TOD_BINS` for the fine bins),
   change `outing_start` and `cat_time_visits` in `schema_v2.py` / `measure_v2.py`, update `fit_v2.py` (the split into fine bins now covers 3 bins) and the ledger caveats (the edges and the choice of 3 are
   public settings chosen after the previous run). Regenerate and commit the artifacts; report SNR per time table.

**3. Richer real-vs-synthetic visual and statistical comparison in `eval/`** (panel list in section 7, item 4). **DONE 2026-11-02** (see section 8; baseline table in `PROCESS.md`).
   Build `src/geolife/eval/viz.py` (plots) and `src/geolife/eval/compare.py` (moments and distances), the notebook `geolife/eval/notebooks/viz_comparison.ipynb`, and a `--viz` switch in
   `geolife/eval/score.py`. It needs real points and synthetic traces side by side, so it is eval-only; add generation into the source city (Beijing) as an option of stage 2 for direct lon/lat overlays.
   Keep it generic over the quantities so new statistics are one-line additions. Record a "before" table (current generator and defaults) in `PROCESS.md` so progress is visible.

**4. Learned parameters are unstable. DONE 2026-11-02 (see `PROCESS.md` steps 4-5 section; D21).**
   On learned artifacts the planted-stats check failed (`stop_revisit_rate`, `users_per_place_mean`, `share_places_ge2_users`) on all seeds, and the three seeds gave
   2.1M-3.8M points for the same 15 users (6M-29M for 50). Fixed by MAP fits of the raw noisy histograms against public priors in `outside/fit_v2.py`
   (plus planted-check semantics aligned to what the tables measure); the planted checks now pass on all three seeds and the 50-user output spread fell from 3.9x to ~1.8x
   (`tests/geolife/outside/test_fit_learned_artifacts.py`).

**5. The generator does not match the real regime. DONE 2026-11-02 (first pass; see `PROCESS.md` steps 4-5 section; D22, D23).**
   Heavier tails (outings per user, active days), longer outings, hub places, users-per-place histogram shape, distinct places per user.
   Closed the biggest gaps with AR(1) receiver noise + in-stop drift, revisit/length/users-per-place calibration and gap-regime changes; before/after table in `PROCESS.md`.
   Remaining gaps (distinct places per user ~2x, points per user ~2-3x, duration median, hub places) are listed there.

**6. Housekeeping. DONE 2026-11-02 (see `PROCESS.md` step-6/D27 section).**
   - v1 code retired: shared helpers moved into the v2 modules (survivor map in `PROCESS.md`), v1 modules and their tests deleted; boundary test passes.
   - Outings parquet export now carries `cells` and `segments` JSON columns, so it feeds the traversal tables (round-trip test added).
   - Full-suite check run with the system python (`/opt/venv`; no `uv` on this machine): geolife + graph_aim + realfake + evaluate all pass;
     `tests/test_run_benchmark.py` needs the heavy synthcity env, still unverified end to end (recorded in `PROCESS.md`).

**7. Privacy hardening. DONE 2026-11-02** (see `PROCESS.md` step-7 section, D21-D23, and the step-6/D27 section). Done: empirical attacks in `eval/`
   (user-level membership inference against the artifact, linkage/re-identification against the synthetic data, seed-averaging
   quantification: `src/geolife/eval/attacks.py`, runner `geolife/eval/run_attacks.py`, report `geolife/eval/reports/privacy-attacks.md`);
   ledger review (`geolife/eval/review_ledger.py`, report `geolife/eval/reports/ledger-review.md`: every stated number verified; caveat
   gaps - analysis region, mode-heuristic cuts, multi-seed replicates - fixed in `measure_v2.CAVEATS` and now baked into the
   regenerated artifacts); artifact signing (optional HMAC-SHA256 over the manifest: `artifact_cli sign/verify --key`,
   `fit_artifacts.py --sign-key`, `generate_synthetic.py --key`; committed artifacts stay valid unsigned); one seed per release
   (`fit_artifacts.py --release`; dev runs with several seeds stamp a replicate caveat); zero-signal budget dropped (D27:
   cat_time_visits, cat_transitions, sensor_step no longer released; rho re-spent via the weighted split; artifacts regenerated,
   SNR before/after in `PROCESS.md`). Note: the attacks were re-run on the 26-table artifacts and the realistic generator
   on 2026-11-02 (`privacy-attacks.md` updated; MI max AUC 0.734 < ceiling 0.769, linkage still at/below chance); the
   comparison reports (`compare_*_steps45`) still describe the superseded 29-table artifacts and are re-run in the next wave.

**8. M last.** After steps 1-5, redo the M work on the improved synthetic data:
   - Re-run M v2 and `geolife/eval/reports/m-v2.md` (the numbers were produced at the old 75 m snap radius and the old generator); re-check the tokenization counts against `outings-eda.md`.
   - Relation significance is too strict in sparse regimes (FDR 0.2 gives 2 edges on the held-out draw; AUC still 0.87): choose the rule on dev only, report on held-out.
   - Map matching is untested on real GPS (tuned on synthetic Gaussian noise; M on real used the cell proxy): run it on a real sample with the Beijing road graph and measure segment agreement.
   - The full real-vs-synthetic transfer test on all users with the road graph: set the acceptance thresholds first (open decision) and declare "M transfers" or not against them.

## 10. Future work (out of scope now, do not forget)

- **Embedding tier:** factorize the noisy co-occurrence (PMI) of coarse cells and categories across users' visit sets (post-processing, no extra budget) so the generator
  reproduces "visited together" structure and interest groups, and teleport can match on embedding similarity. Add only if co-occurrence signal clears the noise.
- **Co-presence in time:** encounters at the same place or segment within a window; needs joint place x time tables and synchronised schedules.
- Adaptive AIM-style query selection, DP top-k vocabulary selection, a Gromov-Wasserstein travel-time term in teleport, finer time bins at a larger budget,
  a second dataset for an independent transfer check, node/edge-level graph variants (see `docs/snap/graph-aim-plan.md`).

## 11. Risks and open decisions

- **Small N:** 139 users; fine places are noise at user level, so common places are coarse plus a few hubs. Roads concentrate traffic and behave better.
- **Map vintage and coverage:** Overture is current, GeoLife is 2007-2012; no work category; only about 30-48% of stops snap to a place. Overture reachability from the
  air-gap is a risk (the map is a cached public parquet fetched on the connected side).
- **Difficulty mismatch:** if synthetic traces are cleaner than real ones M will not transfer; this is why next step 1 is first.
- **Open decisions for the user:** acceptance thresholds for "M transfers" (step 8); the split of the budget between the behaviour tables and the new sensor tables (step 1); the exact 3-bin edges (step 2);
  whether artifacts are published as one seed or several.
