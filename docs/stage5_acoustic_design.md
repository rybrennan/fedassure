# Stage 5 design record: a second modality: underwater acoustics

Why: every stage-4 number is on clothing photographs, and "that is not sonar" is a fair
objection. The instrument's claim is that it does not care what the input is. That claim
should be tested on the kind of input the audience has in mind, without pretending.

## 1. Which data

| Candidate | What it is | Obtainable? | Decision |
|---|---|---|---|
| **DeepShip** (Irfan et al., 2021) | Real hydrophone recordings of ships in the Strait of Georgia, four vessel classes, 47 h total | The public GitHub portion (63 recordings, ~90 min, ~700 MB) clones without any gate; the rest is by email to the authors | **Used, public portion only, unmodified** |
| ShipsEar (Santos-Domínguez et al., 2016) | 90 recordings, 11 vessel types, Atlantic coast of Spain | Full access by email request; samples only on the site | Not used; would need a request |
| UrbanSound8K | 8,732 urban sound clips, 10 classes | Direct Zenodo download when Zenodo is up (it was not, twice, on 2026-09-09) | Fallback, not needed |
| A synthetic "sonar" set we generate | n/a | n/a | **Rejected.** A made-up signal would look like pretending; a known public benchmark is worth more than a plausible fake. |

DeepShip carries no explicit licence on its repository; it is used here for research
measurement, cited, and not redistributed (the `data/` directory is ignored by git).

## 2. How the audio becomes the harness's input

Two-second windows, one-second hop, 28-band log-mel power over 28 frames: a 28×28 image.
The rest of the harness is untouched: same model, battery, faults, statistics, scorer. That is
the point of the exercise, and it is also the honest limit: this is not a sonar classifier,
it is the same small CNN looking at a time-frequency image.

Faults keep a physical reading in that image: bias = DC offset in log power (receiver gain
drift), gain = dynamic-range compression, blur = time-frequency smearing (bandwidth loss),
label noise = a mislabelled contact.

## 3. Split by recording, and the tug problem

Adjacent windows of one recording are near-duplicates. A window-level split would leak the
test set into training and the accuracy would be a lie. The split is **by recording**: the
last fifth of each class's recordings by sorted filename, and at least one, because the public
portion has only four tug recordings. The probe battery is drawn from test windows, so every
class must have test windows; the first split rule left tug with none and was fixed by that
"at least one".

Consequence stated: 52 training recordings, 11 test recordings. Held-out-recording accuracy
is about 53% on four classes with unequal priors. Training loss goes to 0.015, so the model
memorises recordings. That is the dataset's size, not a defect in the harness, and it is why
no claim here is about vessel classification accuracy.

## 4. Five local epochs, not one

With ~450 windows per node, one local epoch is seven optimiser steps per round, thirteen
times fewer than on the clothing set, and at 30 rounds the model was still climbing, which
breaks the stage-2 assumption that rounds 5–14 are a settled reference epoch. Five local
epochs converge by round 10 (test accuracy flat from round 10 to 30). Runtime 3 min per run
against 0.7; the full 99-run grid is 4.9 h. `local_epochs` is in the config fingerprint, so
the two settings never share a family.

## 5. Healthy floor at alpha 0.5, seed 0, 500 probes (for comparison with clothing)

| | clothing | acoustic |
|---|---|---|
| level median | 0.063 | 0.094 |
| across-client SD | 0.021 | 0.048 |
| across-round SD | 0.009 | 0.014 |
| persistence median | 0.39 | 0.47 |
| max within-round z | 3.1 | 5.6 |

Noisier everywhere, as a small, memorised, prior-shifted dataset should be. Persistence sits
at the 1/√5 noise expectation, so the offset-change statistic's healthy assumption holds.

## 6. What the acoustic grid can and cannot say

It can say whether the same ordering of results (label faults localised by persistence,
slow input drift caught by CUSUM under skew, gain the failure case, the fleet metric blind)
appears on real hydrophone data. It cannot say anything about sonar classification, about
a deployment model, or about faults at sea. It is the same instrument on a second, harder,
more relevant input, with its own healthy floors and its own thresholds.
