# Stage 2 design record: probe battery and divergence

Decision record for `telltale/probes.py` and `telltale/detect.py`. The README carries the
current state and results; this file carries the *choices* and the alternatives that were
rejected, so a later stage can revisit one without re-deriving all of them.

Written before implementation, then annotated with what the healthy runs showed.

## 1. What a probe is

A probe is one input image. A **battery** is an ordered set of them, drawn once from a split
no node trains on, fixed for the life of a run, and distributed with the global model. Every
node scores the same battery with its locally trained model and returns the resulting
class-probability rows. Nothing else crosses the interface.

| Choice | Decision | Rejected alternative and why |
|---|---|---|
| Source split | Held-out **test** split | Train split: whichever node holds a probe image in its shard would score it differently for a legitimate reason (it trained on it), confounding the cross-node comparison. |
| Selection | Stratified random, equal per class, seeded by `probe_seed` | Boundary / high-entropy selection is more sensitive to model change, but it is *also* more sensitive to legitimate heterogeneity, which is exactly the confound. Deferred until faults exist to measure it against. |
| Ordering | **Round-robin by class** (probe *i* has class *i mod C*) | Plain shuffle. Round-robin makes every prefix of length *kC* an exactly stratified sub-battery, so one run over a 500-probe battery yields the statistic at every smaller size from the same reports. That is how the bandwidth curve is measured rather than asserted. |
| Labels | Kept server-side only; never sent | Nodes return probabilities, not correctness. Labels are for server diagnostics. |
| Probe images and the evaluation test set | Overlap allowed | Probes are never trained on, so no leakage into the model; the detector trains nothing, so no leakage into detection. Carving them out would shift the baseline accuracy numbers for no gain. |
| Identity | `fingerprint()` = SHA-256 of indices + pixel bytes, echoed in every report | The instrument self-check compares the echoed fingerprint to the served one. |

## 2. Which model scores it

Each node scores the battery with its **post-local-training** state, the same state it returns
for aggregation. That is the object under test: "what this node's model now believes."
The monitor attaches to the existing `update_hook` seam, sees every `ClientUpdate` before
aggregation, and scores each state on a private template model.

**Inertness is asserted, not assumed.** The template is built under `torch.random.fork_rng`,
scoring runs under `no_grad` in eval mode, and a test proves the final parameters are
bitwise identical with and without the monitor attached. At full scale the instrumented
runs are compared against the stored stage-1 baseline accuracy curves.

## 3. What is measured

Per node per round, three quantities, one per handoff bullet:

**Cross-node level.** Mean over probes of the Jensen–Shannon divergence between the node's
row and the **leave-one-out** mean of its peers' rows. JS rather than KL because it is
symmetric and bounded (≤ ln 2), so a saturated probe cannot dominate. Leave-one-out so a node
is never compared against a consensus it contributed to. Within each round the levels are
converted to a robust z-score against that round's own median and MAD: the threshold is set
against the locally measured floor, never an absolute constant.

**Temporal self.** JS between the node's rows this round and its own rows last round. Needs
no peers, so it is the statistic that survives disconnection.

**Offset change and persistence.** The node's *offset* from consensus, d_t = p_t − q_t, is a
vector over (probe, class). A legitimately skewed node has a large offset **that is stable**;
it believes differently every round in the same way. A drifting feed produces an offset that
*moves*. So the candidate discriminator is not the level of d_t but its change against the
node's own trailing mean, c_t = d_t − mean(d_{t−W..t−1}), and specifically whether that change
is **directional across rounds**. Persistence ratio over a window of T rounds:

    ρ = ‖ mean_t c_t ‖ / mean_t ‖ c_t ‖        ∈ [0, 1]

Zero-mean sampling noise gives ρ ≈ 1/√T; a persistent directional shift gives ρ → 1.
Separating those two is the core estimation problem the handoff names.

This is the design reason the *level* statistic alone cannot be the detector: it is confounded
by heterogeneity by construction. Level is still computed and characterised, because it is
the naive detector and its false-alarm behaviour per alpha is a result in its own right.

## 4. Instrument self-check

A degraded scorer and a degraded feed look the same in naive monitoring, so the probe pathway
carries its own checks, separate from the divergence statistics:

- battery fingerprint echoed in each report must equal the served battery;
- report shape must be (n_probes, n_classes), every value finite, every row summing to 1;
- a report bitwise identical to the same node's previous report is a replay: a model that
  trained this round cannot return the same softmax rows;
- scorer determinism: scoring the same state twice must be bitwise equal (a hazard on
  non-deterministic backends; the harness runs CPU).

In the healthy runs the check must report nothing. Stage 3 can break each condition on purpose.

## 5. Bandwidth

Return payload = `n_probes × n_classes × bytes_per_scalar`, independent of model size and
local data volume. Battery size trades sensitivity for bytes. The characterisation splits the
healthy spread of the level statistic into two parts that behave differently with battery size:

- **across rounds within a node**: sampling and training noise, expected to fall as ~1/√n;
- **across nodes within a round**: real heterogeneity plus sampling noise, expected to
  *plateau* at the heterogeneity floor.

The knee where the across-round spread stops falling is the point beyond which more probes
buy no resolution. That knee, per alpha, is the quantitative form of the bandwidth tradeoff.
Quantisation of the payload (float32 → float16 → uint8) is measured as the change it induces
in the level statistic, from the same reports.

## 6. What this stage does not claim

No fault exists, so no detection probability exists. Every number produced here is a property
of the **healthy** federation: the false-alarm side of the operating curve, per alpha and per
battery size. Detection sensitivity is stage 4's to measure, against stage 3's faults.

## 7. Runtime priced before launch

Scoring cost per round is `n_clients` forward passes over `n_probes` images; for 10 clients
and 500 probes, about 5,000 28×28 inferences, negligible against local training. The healthy
characterisation reuses the three stage-1 arms (alpha 0.1 / 0.5 / 100), 30 rounds each,
one train seed: **three runs, ~20 minutes total** on the M4 Pro. No new grid dimension.

## 8. What the healthy runs showed (annotated after the fact)

- Inertness held at scale: all three arms reproduced stage-1 accuracy and loss curves exactly.
- Section 5's prediction was half right. The across-client spread does plateau under skew.
  But the across-round spread does **not** fall as 1/√n: disjoint sub-batteries correlate at
  0.86–0.93 round to round. The temporal floor of the level statistic is training dynamics,
  not probe sampling, so the knee is at or below 50 probes rather than in the hundreds.
- Section 3's within-round robust z is not a transferable threshold. With ten nodes the MAD
  is coarse, and near-IID it is so small that healthy z reaches 6.7. The floor has to be
  measured over a window of rounds. Level was kept and characterised because that failure is
  itself the result.
- Persistence behaved as designed in the healthy state: median at or below 1/√5 at every
  alpha, including severe skew. Its healthy *maximum* is the one statistic that tightens with
  battery size, so battery size is a persistence-ceiling knob, not a level-resolution knob.
- Self-divergence is roughly alpha-independent (0.012–0.014), which was not predicted and
  makes it the better-behaved disconnected-case baseline.
- One instrument fact surfaced by a failing test: CPU convolution output depends on batch
  composition at the 3e-8 level. The scorer now scores the whole battery in one pass.
