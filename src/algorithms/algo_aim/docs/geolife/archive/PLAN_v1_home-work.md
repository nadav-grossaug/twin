# Geo extension: DP map-anchored mobility artifact

Status: draft plan (theory + phasing). Builds on `research/inspectable_synthetic_data_artifact_proposal.docx`.
Example dataset: GeoLife (Beijing GPS trajectories, ~182 users).

## 1. Goal and success criterion

**Setting.** Real trajectories exist only inside the air-gapped network. Only a small, inspectable,
differentially private **artifact** crosses the gap. On the DevVision side we generate **fully synthetic**
users and traces from the artifact, in a *target city of our choosing*, and develop an unsupervised method **M**
there. M finds (a) main places per user, (b) crossings (places/road segments shared by several users),
(c) relations between users that follow from (a) and (b).

**Success = M transfers.** M, developed and tuned only on synthetic data, is run unchanged ("one-shot") on the
real data and behaves the same: it finds main places, crossings and user relations with the same output
distributions and the same error profile. M must be coordinate-agnostic (works on map-relative features,
not absolute lat/lon).

Non-goals: one-to-one correspondence between synthetic and real users (that would be a per-user transform,
not synthesis); a formal guarantee that hides *which* city the data came from (teleportation is
obfuscation of provenance, not DP).

## 2. Where each stage runs

| Stage | Environment | Inputs | Output |
|---|---|---|---|
| Learn (L0-L4) | Air-gapped | Real traces + public map snapshot of the source city (imported) | Artifact (JSON + Parquet/NPZ, checksummed) |
| Review / export | Air-gapped -> DevVision | Artifact | Human-inspected, approved transfer |
| Generate (L5-L7) | DevVision | Artifact + public map of target city (Overture) | Synthetic raw-style GPS traces + ground-truth planted relations |
| Develop M | DevVision | Synthetic data | M (code + config) |
| Validate | Air-gapped | M + real data | M's real-data output vs. synthetic error profile |

The artifact is the only data that crosses outward; M is the only thing that crosses inward.

## 3. Privacy model

- **Protected unit:** a user (all their trajectories). Contributions are capped per user.
- **Why it matters:** in mobility data the locations *are* the identifiers (a few points single out a person).
  Encoding places as tokens does not remove this: the map is public and tokens can be inverted. The
  artifact therefore holds only DP aggregate statistics, never real IDs, coordinates or traces.
- **Mechanism:** Gaussian noise on clipped per-user count vectors, zCDP accounting; all steps after
  measurement (embedding, model fitting, generation, teleport) are post-processing.
- **Strength:** moderate is acceptable (target epsilon range to be fixed, initial guess 4-8) because the
  released object is a low-dimensional aggregate. Empirical linkage/membership attacks are run as a sanity check,
  not as a substitute for the guarantee.
- **Public vocabulary:** candidate places and roads come from the public map, so vocabulary selection leaks nothing.
  Deeper hierarchy levels are released only when noisy counts clear a threshold.
- **Random IDs:** per-user pseudonyms and latent place IDs are used for hygiene; they are not a privacy mechanism.

## 4. Core idea: behaviour vs. geography

- **Behaviour (transferable, stored in artifact):** abstract place roles/categories, popularity and
  user-overlap structure, dwell times, hour/day profiles, OD flows, travel mode, route-choice statistics.
- **Geography (not transferable, regenerated):** concrete POIs and road segments in the target city; routes are
  recomputed by routing on the target road graph, not copied.

Crossings emerge from shared anchors plus routing, so overlap between users must be modelled explicitly;
independent per-user sampling would erase it.

## 5. Pipeline layers

| Layer | Content | Privacy cost |
|---|---|---|
| L0 Public map | Overture transportation (segments + connectors), Places + category taxonomy, buildings, H3 index | none |
| L1 Per-user preprocessing | HMM map-matching to a road path; stay-point detection snapped to a place/H3 cell; trip segmentation; pseudonymize; contribution caps | none if bounded |
| L2 Tokenization | Road tokens (class, segment), place tokens (category, cell, id), relational features (road class, turn type, speed vs. limit, land use, hour) | none |
| L3 DP measurement | Hierarchical noisy counts: distinct users per place, dwell, place-to-place flows, place-hour profile, road usage, detour ratio / road-class mix, co-visit and shared-anchor structure, users-per-place histogram | Gaussian, zCDP |
| L4 Embedding | Factorization of noisy co-occurrence/PMI (structural tier, transferable) + public map-derived embeddings; identity tier (segment/POI ids) only for same-city evaluation | post-processing |
| L5 Generative model | Population -> user routines (mixture) -> semi-Markov schedule over latent places -> trips -> route. Parametric prior (Zipf popularity, exploration/preferential return, gravity OD, near-shortest-path) with embedding tier adding capacity where budget allows | post-processing |
| L6 Teleport / grounding | Assignment of latent places to target-city POIs (match on category, popularity percentile, local density; optional travel-time consistency via Gromov-Wasserstein); consistent across users; homes sampled from building footprints | post-processing |
| L7 Render | Routing on target graph; emit **raw-style GPS** (sensor noise, sampling gaps, dwell jitter, multi-modal trips) so M's full pipeline (map matching, stay detection) is exercised; planted ground-truth relations (shared workplaces, co-travel, hubs) | post-processing |

## 6. Artifact layout (extends the proposal)

```
artifact/
  manifest.json        # format version, learner, hash, source-map snapshot id
  schema.json          # token types, hierarchy, categories (no real ids/coords)
  preprocessing.json   # grid/H3 resolution, time bins, stay-point + matching parameters
  model/
    behaviour/         # popularity, flows, dwell, hour profiles, overlap structure
    embeddings/        # structural embeddings (npz) + metadata
  privacy.json         # adjacency (user), caps, epsilon/delta or rho, accountant, assumptions
  grounding.json       # spec for teleport (category mix, density percentiles, scale)
  diagnostics.json     # fit checks, marginal errors
  generation.json      # sampling settings, seeds, post-processing
```

No pickle. Clean-room test: copy only `artifact/` to a fresh environment and generate.

## 7. Evaluation

1. **M-transfer (primary):** run M on synthetic and on real; compare output distributions and error profile
   (precision/recall of main places, false-crossing rate, relation-graph statistics, stability under resampling).
   On synthetic data the planted ground truth gives exact scores; on real data only distributions can be compared.
2. **Distribution fidelity:** users-per-place histogram, place popularity rank correlation, encounter rates,
   user-user co-visit graph degree distribution, trip length/duration, radius of gyration, OD flow error.
3. **Same-city check:** artifact -> generate back into the source city (identity tier), compare against ground truth.
4. **Transfer check:** generate into a different city; confirm M's behaviour is unchanged (no real ground truth; compare EDAs).
5. **Privacy sanity:** linkage and membership-inference attacks on the generated data.

## 8. Phases

- **P0 Feasibility:** Overture reachability from DevVision and the air-gap import path; routable graph from
  Overture connectors+segments; map-matcher choice; datum check (WGS-84 vs GCJ-02); GeoLife access;
  confirm installed Synthcity version/plugins (see `uv.lock`).
- **P1 Substrate:** L1-L2 on GeoLife (map matching, stay points, tokenization) + EDA notebook.
- **P2 Non-private artifact:** parametric behaviour model + inspector + clean-room generation in the *same* city.
- **P3 DP measurement:** L3 with contribution caps and accountant; hierarchical release; embeddings (L4).
- **P4 Teleport + render:** L6-L7 into a target city; planted relations.
- **P5 M and transfer evaluation:** build M on synthetic data; error-profile comparison against real.
- **P6 Hardening:** privacy attacks, artifact signing/versioning.

## 9. Risks

- **Small N (~182 users):** at user-level DP only places visited by tens of users survive; use a coarse
  hierarchy and expect weak signal for fine-grained crossings. 
- **Map vintage:** Overture is current, GeoLife is 2007-2012; POIs and roads differ.
- **Difficulty mismatch:** if synthetic traces are cleaner than real ones, M will not transfer; render noise must
  be calibrated from DP-released noise statistics.
- **Relational fingerprints:** exported place-place relations could allow re-location by graph matching;
  release only coarse, noised relations.
- **Overlap structure is the hard part:** user-overlap (shared anchors, co-visits) must be measured and
  reproduced explicitly, within the contribution caps.
- **Tooling/network:** Overture hosting (S3/Azure) may be firewalled; map-matching tools need routable data.

## 10. Decisions

1. **Privacy unit:** user-level. Start there; scale back (coarser hierarchy, larger epsilon, tighter caps) only if
   the signal does not survive, and record the change in `docs/PROCESS.md`.
2. **No second dataset.** Transfer is checked by comparing EDAs of the real GeoLife data (inside the
   pretend air-gap) with the synthetic data (outside). The "different city" generation test stays, but there is no
   independent real ground truth for it.
3. **Code location:** `src/geolife/` (keeps the merge clean). Tests in `tests/geolife/`.
4. **Generation-time target city:** chosen at generation time (assumed).

## 11. Still open

- M's definition of "relation" between users (shared anchor, co-presence window, co-travel); needed to fix the
  L3 query workload.
- Target epsilon value (initial guess 4-8).
- Whether Adult is used to validate the artifact contract first (default: no, GeoLife only).
