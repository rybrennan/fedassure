# Changelog

Stages are tagged. Every tag's README states what that stage claimed and what it did not.

## v0.5.0 — 2026-09-09 — acoustic grid scored; node confound resolved; submission hygiene
- DeepShip grid (99 runs): the ordering by heterogeneity transfers; CUSUM transfers,
  persistence does not; gain is caught here; the fleet metric is uninformative rather than
  hidden; severe skew catches nothing.
- Node dimension (72 runs): node-independent at near-IID and moderate skew; node-dependent
  at severe skew.
- Licence, citation, lock file, CI, Dockerfile, reproduce script, tracked result summaries.

## v0.4.0 — 2026-09-09 — second modality, submarine schedule, threshold sweep
- DeepShip underwater acoustics through the unchanged harness (`fedassure/acoustic.py`).
- Contact schedules: submarine nodes report every k rounds; temporal statistics run over a
  node's own contacts; time to detection in contacts.
- Thresholds per platform class; CUSUM reference spread pooled where a class is starved.
- Two-sided CUSUM judged on healthy runs; threshold sweep per statistic per alpha;
  `results/scores.json` carries every number in the README's tables.
- Overnight grids: node dimension (clothing) and the full fault grid (acoustics).

## v0.3.0 — 2026-09-09 — stage 4: the grid
- 99-run grid (3 alphas × 3 seeds × healthy + 10 faults), 10 h, scored into detection
  probability with Wilson intervals and a chance floor; false-alarm rate on held-out seeds.
- Fleet metric hid all 90 faults; label corruption localised at every alpha; slow drift under
  skew caught by CUSUM alone; gain the failure case.

## v0.2.0 — 2026-09-08 — stages 2 and 3: instrument and faults
- Probe battery, monitor on the `update_hook` seam proven bitwise inert at scale; healthy
  divergence floor per alpha and battery size; bandwidth knee at ~1 kB.
- Fault injection with known ground truth through a second seam; metrics for Pd / FAR / ttd.

## v0.1.0 — 2026-09-07 — stage 1: clean baseline
- FedAvg implemented directly, Dirichlet non-IID partitioning, seeds split by concern,
  bitwise reproducibility asserted. Three alpha arms; the noise floor scales with
  heterogeneity 3.4×. No detector, no faults, no detection claim.
