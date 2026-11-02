# Geo plan v2: outings and visited places (no home/work)

Status: draft. Supersedes the home/work and EPR-anchor parts of `docs/geolife/PLAN.md`; everything else
(air-gap flow, user-level DP, public-map vocabulary, artifact format, teleport, M-transfer criterion) stays.

## 1. Why change

The first real run (25 users) showed that M's home/work roles almost never fire (4% home, 1% work, 94% other) and that
only 8% of users get a relation, while the synthetic data produced 72%. GeoLife is mostly outings: trajectories to and
from places visited, observed over weeks to months per user, with no stable overnight home or workday routine. The
generator's model (anchors, daily schedules, overnight stays) describes a different kind of data.

New view of the data: **a user is a set of outings; an outing is a trajectory with a few stops.** What generalises across
users and cities is *where outings go* (which places and road segments, how popular), *what they look like* (stops,
duration, length, mode, time of day), and *who shares them*.

## 2. What the downstream task becomes

| Output of M | Definition (v2) |
|---|---|
| Places of interest | Stops (dwell above a threshold) clustered and snapped to map places; ranked by distinct outings and distinct days |
| Common places | Places visited by at least k distinct users |
| Crossings | Road segments / intersections traversed by at least k distinct users (map-matched routes) |
| User relations | Co-visit graph: users linked by significantly shared places or segments. Edge weight is rarity-weighted (IDF) and tested against a null model that uses place popularity; communities on top |

Dropped: home/work roles, anchor counts, return-to-home logic. Co-presence in time stays out of scope (later tier).

## 3. Changes by layer

| Layer | Change |
|---|---|
| Contract (`types.py`) | Add `Outing` (user, t_start, t_end, mode, ordered stop place ids, matched segments, length) and `Visit` (place, category, cell, arrival, dwell). Keep `Stay`/`Trip` so v1 code and tests still run. Orchestrator-owned. |
| Tokenization | Each trajectory (and any gap above a threshold, e.g. 2 h) is one outing; stops inside it are visits; start and end places are visits. Mode from labels or speed. |
| Caps (user-level DP) | Cap per user: outings, distinct places, distinct segments. Sensitivity is computed from the caps, as now. |
| DP measurements | (a) distinct users per place, hierarchical (H3 6 -> 7 -> 8 -> place) as in v1; (b) category x hour-of-day and day-of-week; (c) outing shape: stops per outing, duration, length, mode; (d) stop-to-stop transitions at category and coarse-cell level; (e) road-class and segment usage (distinct users per segment, hierarchical by cell); (f) per-user visit-set size distribution and revisit rate; (g) overlap structure: histogram of users per place (kept) plus co-visit degree histogram under a per-user neighbour cap. |
| Embedding (optional tier) | Co-occurrence / PMI of coarse cells across users' visit sets (post-processing of (a)/(d)), used for similarity between cells and for group profiles. |
| Generator | Replace anchors + schedules by an **outing process**: each user has a small activity region and a taste vector over categories; an outing draws a start, a number of stops, and each stop by popularity x taste x distance decay; revisit probability gives favourite places without roles. User groups (shared interest regions, e.g. a campus) are latent and drive shared places; some pairs co-travel. Planted truth: groups, shared places, co-traversed segments. |
| Render | Sparse, bursty logging: active-days window per user (weeks to months), gaps, variable sampling, mode-dependent speeds; sensor parameters taken from the artifact `generation` dict. |
| Teleport | Unchanged in principle: ground latent places by category + popularity percentile + density; homes are no longer needed, so the weak residential/office grounding stops mattering. |
| M | No roles. Places of interest from stop clusters; common places and crossings by distinct-user counts; relation graph from rarity-weighted co-visits with a significance test; communities. Map-relative features only. |

## 4. Evaluation (works without real labels)

1. **Planted-truth recovery on synthetic data:** precision/recall of places of interest, common places (k = 2, 3, 5), crossings,
   relation edges, community ARI. Sweep noise, sampling, gaps, window length, user count (as in v1).
2. **Real-vs-synthetic distributions (the transfer test):** places per user, outings per user, stops per outing, outing length and
   duration, revisit rate, users-per-place histogram, co-visit graph degree, users-per-segment histogram. Compare with
   total variation; the sensor regime of the synthetic render is matched first from the released generation parameters.
3. **Real-data self-consistency (label-free, new):** on real data, split each user's outings into two halves and check that M's
   places, common places and relations reproduce across halves (split-half reliability), and that held-out visits are
   predicted by the co-visit graph. If M is not reliable on real data against itself, a synthetic match means nothing.
4. **Privacy:** user-level ledger with caveats; empirical linkage/membership attacks on generated outings.

## 5. Phases and agent tasks

| # | Task | Output | Depends on |
|---|---|---|---|
| T0 | Outing EDA on all 182 users: outings per user, stops per outing, revisit rate, active-days span, distinct places per user, co-visit overlap at several resolutions, sampling regime | `geolife/inside/notebooks/02_outings_eda.ipynb`, report | none (use real data, in the pretend air-gap) |
| T1 | Contract v2 (`Outing`, `Visit`), tokenization into outings | `types.py` additions, `tokenize/outings.py`, tests | T0 |
| T2 | DP learner v2: new measurement set, caps, weighted budget with a floor, hierarchy fixed from public choices before looking at SNR | `learn.py`, `generate/measure.py`, SNR report | T1 |
| T3 | Generator v2: outing process, groups, planted truth, bursty render | `generate/` additions, tests, fake artifact | T2 (schema) |
| T4 | M v2 | `m/` changes, tests, planted-truth recovery report | T3 |
| T5 | Transfer evaluation on all 182 users with the road graph: distributions, split-half reliability, sensor-regime matching | report | T0, T2, T4 |
| T6 | Hardening: attacks, ledger review, unaccounted-tuning audit, signing | report | T5 |

T0 comes first and decides parameter ranges; T3 and T4 can overlap once the artifact schema from T2 is fixed.

## 6. Risks and open decisions

- **Uneven observation:** users range from days to years of data; per-user caps and a minimum-activity rule are needed, and both are public choices.
- **Weak overlap at fine scale:** place-level overlap was already noise at H3 res 8 at user-level DP; common places will be coarse (cell or district) plus a few hubs. Crossings by segment may behave better since roads concentrate traffic; T0/T2 will tell.
- **Outing definition:** gap and dwell thresholds shape everything downstream; they should be fixed from public reasoning and the EDA, not tuned on M's output.
- **Co-visit significance:** the null model must use only released popularity statistics so that M runs identically on synthetic and real data.

## 7. Decisions (user, 2026-11-01)

1. **Time of day and day of week are in scope.** They are features for M (visit-time profiles per place and per user) and are
   measured in the DP learner (category x hour, place-tier x hour / day-of-week, outing start-time and weekday/weekend mix) and
   modelled in the generator. Cost: temporal tables are sparser, so they use coarse bins (e.g. 4 time-of-day bins x weekday/weekend,
   refined only where counts clear the noise).
2. **No co-presence now.** Relations are co-visit only (shared places/segments, optionally with similar visit-time profile).
3. **Parametric first.** The artifact holds fitted tables and exponents; no embedding tier in this version.
4. Epsilon stays about 6 at user level unless the T2 report says otherwise.

## 8. Future work (out of scope now, do not forget)

- **Embedding tier:** factorize the noisy co-occurrence / PMI of coarse cells and categories across users' visit sets
  (post-processing, no extra privacy cost) so the generator can reproduce "visited together" structure and interest groups, and so
  teleport grounding can match on embedding similarity. Add only if the T2 SNR report shows co-occurrence signal above the noise.
- **Co-presence in time:** encounters at the same place or segment within a time window; needs joint place x time measurements and
  synchronised schedules in the generator.
- Adaptive AIM-style query selection, DP top-k vocabulary selection, Gromov-Wasserstein travel-time term in teleport, and node/edge
  graph extensions (see `docs/snap/graph-aim-plan.md`).
