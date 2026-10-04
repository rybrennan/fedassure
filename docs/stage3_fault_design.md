# Stage 3 design record: fault injection and the first operating measurements

Decision record for `telltale/faults.py`, `telltale/metrics.py`, and the statistics added
to `detect.py` during stage 3. The README carries current results; this file carries the
choices, what motivated each, and which were made *after* seeing a fault run.

## 1. The seam

A second hook on `run_federated`, `fault(client_id, round_idx, x, y) -> (x, y)`, applied to a
client's shard before local training. The federation code knows nothing about faults.

| Choice | Decision | Rejected alternative and why |
|---|---|---|
| Where | Per-round transform inside the loop | A wrapped `Dataset` cannot ramp over rounds. |
| Inertness | A fault before onset, or at severity zero, returns the *same tensor objects* | Cloning would still be numerically identical but the bitwise test is stronger and free. |
| Determinism | Label noise draws from `(seed, node)` via a private generator | Global RNG would perturb every other client's batch order. |
| Composition | One spec per node, specs on different nodes compose | Enough for a grid; multi-fault nodes are not in scope. |

## 2. Fault kinds and the ramp

Magnitude rises linearly from zero over `ramp` rounds from `onset`; `ramp=0` is a step. Four
kinds in the normalised input space the model sees: additive `bias`, multiplicative `gain`,
Gaussian `blur`, and `label_noise` with a nested corruption set so a ramp is monotone.

**What the first ladder showed about additive bias.** At alpha 0.5, node 2 (6% of fleet
weight), ramps to 0.5, 1.0 and 2.0 standard deviations all produced the *same* signature:
about 0.02 nats between the node's faulted and healthy rows, a 2–4 point drop in the node's
own probe accuracy, and a *rise* in its confidence. A convolutional net with learnable bias
terms absorbs a constant offset within one local epoch. Additive DC drift is therefore a fault
the model self-corrects, and its severity ladder is flat. It stays in the set as the weak case;
blur and gain, which change information content rather than offset, carry the ladder.

## 3. Statistics added during stage 3, and why that is stated here

The stage-2 design held that a legitimately skewed node has a large offset that is *stable*,
and that change against the node's own history is the discriminator. Two statistics were added
after fault runs, each an application of that principle to the level, and each with its
threshold still taken from healthy runs only:

- **Self-referenced level** (own-history z over a 5-round window), after the 0.5σ bias
  *ramp* doubled the node's level while its within-round z *fell*, because the fleet
  threshold is set by the most-skewed healthy node and the drift moved the target toward
  the fleet median. Catches steps at onset; misses ramps, whose per-round increment sits
  inside the node's own noise.
- **CUSUM** (Page 1954, allowance 0.5 SD, reference epoch rounds 5–14 fixed for every node
  and run), after the step/ramp asymmetry above. Catches the ramp at 30% of final severity.

A third was considered and rejected: tuning the CUSUM allowance or reference epoch to the
runs. The allowance is the textbook default and the reference epoch is the settled part of
the burn-in, chosen once.

## 4. What the healthy runs say about CUSUM, and the per-alpha rule again

The healthy CUSUM ceiling is 0.7 / 1.1 / 19.7 at alpha 0.1 / 0.5 / 100. Near-IID the
reference-epoch SD is tiny (0.003 nats) and one healthy node's level rose 0.004 nats over
fifteen rounds, a 36% climb in a fleet where everyone agrees, accumulated as 1.4 SD per
round. That is a healthy node legitimately wandering. The consequence is the stage-1 rule on
a new statistic: the threshold must be per alpha, and a fault near-IID has to beat 19.7 where
under skew it has to beat 1.1. Whether it does is measured, not asserted.

## 5. Thresholds and the false-alarm rate

`metrics.calibrate` takes the healthy maximum after a 10-round burn-in. On the calibration
seed that gives a false-alarm rate of exactly zero by construction, which is not a
measurement. Healthy runs on other train seeds are held out and the false-alarm rate is
counted there, per alpha, per statistic (`scripts/score_faults.py`).

## 6. Grid priced

Every run is 6.5 minutes. Today's set: three healthy arms, five fault runs at alpha 0.5,
four held-out healthy seeds, two fault runs at the other alphas, two more fault kinds:
sixteen runs, under two hours, all on one laptop core set, queued sequentially so the
thread count stays identical to the stage-1 baseline (bitwise comparison requires it).

## 7. Contact schedules: the submarine case

Added after the brief was re-read: section 2.4 already claims tolerance of intermittent
connectivity and that temporal self-divergence works disconnected. Neither had been run.

| Choice | Decision | Rejected alternative and why |
|---|---|---|
| Seam | `participation(round_idx) -> [client ids]` on `run_federated`, a third seam next to the hook and the fault | Adding schedule fields to `FedConfig` would change every existing fingerprint. |
| Silent-round model | The boat does nothing: no training, no report; on its next contact it syncs to the current global model | The realistic alternative: the boat keeps training on a stale model while submerged and returns one large update on surfacing, is asynchronous aggregation (FedBuff-style). It changes the aggregation rule itself, which is a research question in its own right and would confound every integrity measurement with a staleness effect. Out of scope for this harness; named in the handoff as the phase-two extension. |
| Statistics | Every temporal statistic runs over a node's own observed rounds | Computing against fleet rounds made a silent round look like a step. |
| Timing | Time to detection is also counted in contacts | "Five rounds" is meaningless to a boat that surfaces every third round. |
| Downlink | Not modelled, because the battery is fixed for the life of a deployment: it is loaded before the patrol and only its fingerprint travels | Sending five hundred samples down a submarine link would cost more than every report the boat ever sends back. The stable-battery design already avoids it; this is stated rather than simulated. |

What the store-and-forward extension would look like, for the record: a submerged boat could
score the fixed battery against its own evolving local model at every local training pass,
keep the trajectory (a kilobyte per pass), and transmit the whole curve on surfacing. The
temporal self-divergence statistic is exactly that computation. It needs the stale-training
model above to simulate honestly, so it waits for the same phase-two work.
