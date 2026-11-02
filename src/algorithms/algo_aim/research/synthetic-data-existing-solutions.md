# Synthetic data generation: existing out-of-the-box solutions

Scope: tabular and time-series data, with a focus on differential privacy (DP).
Date: 2026-10-14.

Verification: version and license facts below were read from PyPI/GitHub pages during research. Items marked **(unverified)** were not confirmed. GitHub pages gave no commit dates, so "maintenance" is inferred from PyPI release dates. Re-check licenses before use.

DevVision note: install anything through the JFrog PyPI proxy (via `uv`). Many research repos below are not on PyPI and would need to be vetted and mirrored first.

## Summary

| Tool | Tabular / Time series | DP | License | Latest release | pip | Built-in evaluation |
|---|---|---|---|---|---|---|
| Synthcity | Tab, TS, survival | PrivBayes, DP-GAN, PATE-GAN | Apache-2.0 | 0.2.12 (2025-05-08) | yes | extensive |
| SmartNoise Synth | Tab | true DP (epsilon budget) | MIT | 1.0.8 (2026-04-15) | yes | partial |
| dpmm (SAS) | Tab | true DP, end-to-end (MST, AIM, PrivBayes+PGM) | Apache-2.0 | (unverified) | yes | notebooks only |
| mostlyai | Tab, sequential, multi-table | DP training option (guarantee unverified) | Apache-2.0 | 6.2.0 | yes | HTML QA reports |
| SDV | Tab, multi-table (HMA), sequential (PAR) | none found | BUSL-1.1 (not open source) | 1.38.5 | yes | quality + diagnostic reports |
| ydata-synthetic | Tab, TS | none | MIT | (unverified) | see caveat | no |
| ydata-sdk | Tab, TS | (unverified) | custom, restrictive | 3.2.3 | yes | (unverified) |
| gretel-synthetics | Tab (ACTGAN), TS (DGAN) | (unverified) | source-available, restricted | 0.22.20 (2025-06-24) | yes | (unverified) |
| DataSynthesizer | Tab | DP claimed | MIT | (unverified) | (unverified) | no |

Building blocks (not synthesizers): Opacus (DP-SGD, Apache-2.0, 1.6.0, 2026-05-05), diffprivlib (MIT, 0.6.6), OpenDP (MIT), private-pgm / mbi (Apache-2.0).

Research code, not packaged: TabDDPM, CTAB-GAN+, TabSyn, TimeGAN, TimeVAE, DoppelGANger, be-great (pip, MIT, 0.0.14, LLM-based, no DP).

## Tabular

**Synthcity** (<https://github.com/vanderschaarlab/synthcity>)
- Best single framework for a bake-off. Includes CTGAN, TVAE, normalizing flows, ADS-GAN, bayesian_network, PrivBayes, DP-GAN, PATE-GAN, TabDDPM, GReaT, ARF, DECAF, GOGGLE.
- Broadest evaluation suite: sanity checks, KS/chi-squared, JS and Wasserstein distance, alpha-precision, detection and performance (TSTR-style) metrics, and privacy metrics (k-anonymity, identifiability, re-identification).
- Caveat: last release was May 2025, so it is slowing down. Dependency pins may conflict with a recent torch.

**SmartNoise Synth** (<https://github.com/opendp/smartnoise-sdk>)
- Actively maintained, MIT, formal DP with an epsilon budget. README confirms MWEM and PATE-CTGAN. MST, AIM and DP-CTGAN are believed to exist **(unverified)**.
- The most credible "real DP" production-style option for tabular data.

**dpmm** (<https://github.com/sassoftware/dpmm>)
- MST, AIM and PrivBayes+PGM with DP domain extraction and DP binning, so the pipeline is DP end to end (many tools only make the model DP and leave preprocessing outside the budget).
- Authors describe it as research-only, not for production. Python 3.10/3.11 only. Small project (17 stars at time of research). Paper: arXiv 2506.00322.

**SDV** (<https://github.com/sdv-dev/SDV>)
- GaussianCopula, CTGAN, TVAE, multi-table HMA, quality and diagnostic reports. Easiest to use.
- License is BUSL-1.1, which is source-available with restrictions. Get a license review before production or commercial use. No DP found.

**mostlyai** (<https://github.com/mostly-ai/mostlyai>)
- Apache-2.0, runs locally, TabularARGN model, HTML QA reports with fidelity and privacy metrics. DP is listed as a training option. Check the docs for whether it gives a formal (epsilon, delta) guarantee.

**Others**
- *ydata-synthetic* (MIT): GAN, WGAN-GP, CTGAN, Gaussian Mixture, no DP. The README says it was renamed, so verify the package before installing. *ydata-sdk* has a restrictive custom license, so avoid it.
- *gretel-synthetics*: GitHub repo archived 2026-02-18, restricted license, pins conflict with SDV and torch. Do not start new work on it.
- *DataSynthesizer* (MIT): legacy PrivBayes-style research tool.
- *TabDDPM*, *CTAB-GAN+*, *TabSyn*, *be-great*: strong non-DP fidelity baselines. TabDDPM and GReaT are wrapped by Synthcity, which is the easiest way to run them.

## Time series

Time series has much weaker out-of-the-box support, and there is no turnkey DP time-series package.

- **mostlyai**: sequential data (multiple records per subject), Apache-2.0, pip. The most usable permissive option for event or sequence data.
- **Synthcity**: TimeGAN and Fourier Flows. The README mentions TimeVAE, but it is not in the methods table.
- **SDV PAR**: sequential model, license caveat above.
- **ydata-synthetic**: TimeGAN and DoppelGANger, no DP, rename caveat.
- **TimeVAE** (<https://github.com/abudesai/timeVAE>, MIT): research code, no pip package.
- **DoppelGANger** (<https://github.com/fjxmlzn/DoppelGANger>): includes a DP variant via TensorFlow Privacy, but needs TensorFlow 1.x, which is outdated.
- **DGAN (gretel-synthetics)**: PyTorch DoppelGANger, but see the gretel caveats.
- **DP for time series in practice**: build your own model (for example a VAE) with Opacus. This also matches the "DP encoder/decoder" idea.

## Evaluation tools

- Synthcity metrics (above) and SDV/SDMetrics (<https://github.com/sdv-dev/SDMetrics>) are usable on any real/synthetic pair.
- SDNist (<https://github.com/usnistgov/SDNist>, v2.5): HTML utility and privacy report, but only against bundled NIST ACS benchmark data, so not a general evaluator.

## Recommendation

1. **Tabular with DP**: start with SmartNoise Synth, add dpmm (MST/AIM) for comparison, and use Synthcity for PrivBayes, DP-GAN and PATE-GAN.
2. **Tabular non-DP baselines**: Synthcity (CTGAN, TVAE, TabDDPM), plus SDV if the license is acceptable.
3. **Time series**: mostlyai sequential or Synthcity TimeGAN as non-DP baselines. Expect to build the DP version yourself with Opacus.
4. **Evaluate everything with one common harness**, so the numbers are comparable (see `synthetic-data-evaluation` ideas: fidelity, TSTR vs TRTR, privacy attacks, a privacy-utility curve over epsilon).
5. **Avoid**: gretel-synthetics, ydata-sdk.

## Open questions

- Does mostlyai DP give a formal (epsilon, delta) guarantee?
- Which of MST, AIM and DP-CTGAN does the current SmartNoise Synth release expose?
- Are PAR and HMA fully available in the open SDV package?
- Licenses of CTAB-GAN+, TabSyn, SDNist and TimeGAN.
- Which of these are mirrored in, or can be added to, JFrog.
