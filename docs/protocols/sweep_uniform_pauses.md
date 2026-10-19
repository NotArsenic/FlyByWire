# Protocol: uniform holding headings with time-freeze pauses

Written 2026-10-19 18:50 +0530, before any data for this sweep was collected. Edits after this point
are dated amendments at the bottom; the text above them is not changed.

D37 and D39 found a multi-stable heading code: under a held frame the decoded heading drifts toward
stable points near 0 and +-85 deg, the headings the turning expert rests at (corridor heading and
the far end of its 90/270 deg excursions). Pause training (D39) flattened the velocity field about
2x but kept the stable points. This sweep asks whether the stable points come from where the
expert rests, and whether real and rewired wiring separate once the task asks for a heading held
anywhere on the circle. It is the last real-vs-rewired comparison of the project; whatever it
gives is reported as is.

## Hypotheses

- **H1 (mechanism).** Holding headings drawn uniformly from the circle flattens the velocity field
  and removes the stable points near 0 and +-85 deg, compared with the same training on the old
  maneuvers.
- **H2 (wiring).** With uniform holds, the real mask beats the rewired one on pause drift,
  velocity-field rms and bump share.
- **H3 (confirmation).** D38's post hoc finding that the rewired mask flies better in closed loop
  replicates on fresh seeds.

## Data

- Uniform expert: `expert.maneuvers.targets=uniform` (far heading drawn from U[-180, 180) deg,
  reached and left the short way), `hold_s=[1.0, 3.0]`, `wait_s=[0.5, 2.0]`, DART noise as D38
  (`sd [0.1, 0.25, 0, 0.8]`, `tau_s 0.5`, `scale [0, 1]`), everything else default. Training set
  `data/expert_uniform`, seeds 0-999 (the same corridors as `data/expert_dart`); test set
  `data/expert_uniform_test`, seeds 2000-2099. No DAgger: D38 did not show it helps beyond DART.
- Control: `data/expert_dart` (seeds 0-999, the old maneuvers, same DART noise), its cache rebuilt
  as `data/cache_dart` (deleted in the cleanup before this sweep).
- A 48-episode pilot of the uniform expert (seeds 0-47, no collisions) was used only to pick
  `wait_s` and `hold_s`; it was scored on heading occupancy, not on any network.
- Pause banks, one per training cache: `--per-episode 4 --hold 100 300 --recovery 60` (1-3 s
  holds; D39 used 0.48-1 s). Test banks on `data/cache_uniform_test`: `--hold 100 100` for drift
  and the spliced scores, and a dense one (`--hold 100 100 --per-episode 27 --recovery 8`) for the
  velocity field.

## Runs

Anatomical readout, `bc.pauses={bank: <arm's bank>, per_episode: 2, p: 1.0}`, 50 epochs,
`model_best.pt` (best validation heading R^2), config otherwise default. Seeds 10-15 have never been
used by any run in this repo.

- Uniform arm: real and rewired, seeds 10-15 (12 runs), trained on `data/cache_uniform`.
- Control arm: real and rewired, seeds 10-12 (6 runs), trained on `data/cache_dart`.

Order: uniform s10-12, control s10-12, uniform s13-15; at most three trainings at a time. A run
that crashes or diverges is rerun with the same seed; seeds are never swapped or dropped.

All scoring is on the uniform test set, for both arms, so the velocity field has samples at every
heading.

## Metrics

From `scripts/eval_pauses.py` (with the test bank and the dense drift bank) and
`scripts/gate3_heading_decode.py`, one value per run:

- pause drift: mean |drift| of the decoded heading 2 s into a held frame (real transient
  included), `pause_eval.json`;
- velocity-field rms: root mean square over 20 deg start-heading bins of the median drift over the
  next second, deg/s (as in `scripts/summarize_pauses.py`), and its stable points (downward zero
  crossings);
- bump share: Gate 3 `ring.anatomical.fraction_mean`, 0.5 threshold unchanged.

Closed loop: every run of the uniform arm, 200 episodes, seeds 10000-10199, `scripts/failure_breakdown.py fly`,
success = no collision before the 10 s limit.

## Tests

All tests are exact permutation tests over relabelings of runs (the run, i.e. the seed, is the
unit), two-sided, p = fraction of relabelings with |difference of means| >= the observed one.

- H1: field rms, uniform vs control, mask labels kept (relabel arm within each mask:
  C(9,3)^2 = 7,056). Supported if p < 0.05 with the uniform arm lower **and** a stable point within
  20 deg of +85 or -85 deg appears in fewer than half of the uniform runs. The stable points near
  0 deg are reported the same way, descriptively: the corridor heading is still where the expert
  spends most of its time.
- H2: real vs rewired inside the uniform arm, 6 vs 6 (C(12,6) = 924), on pause drift, field rms and
  bump share, Holm-corrected over the three. Supported if at least one survives correction in the
  real-better direction (lower drift, lower rms, higher share) and none goes significantly the
  other way.
- H3: closed-loop success, real vs rewired in the uniform arm, 6 vs 6. Replicates if p < 0.05 with
  rewired higher.

Everything else (heading R^2, spliced-pause scores, held-frame and grey drift) is descriptive.
No other metric is added after the fact.

## Reporting

Every per-run value, mean, sd, min and max per cell, and every p, in
`runs/sweep_uniform/stats.json`, and a DECISIONS entry with the verdict on each hypothesis.

## Execution note (2026-10-20, not an amendment)

Claude Code stopped the sweep and a concurrent PPO run when the machine ran low on memory, with 7 of
18 runs trained. The sweep was restarted alone; finished runs were kept, interrupted ones retrained
from scratch with the same seed (training is deterministic). The pause evaluation of
uniform_rewire_s10 had failed on CUDA OOM while post.sh marked the run done; it was rerun with the
same command before scripts/sweep_uniform_stats.py was run. Nothing else changed.
