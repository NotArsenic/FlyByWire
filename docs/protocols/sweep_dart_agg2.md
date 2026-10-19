# Protocol: real vs rewired wiring on the DART + DAgger data

Written 2026-10-19 08:10 +0530, before any run of this sweep was started. Edits after this point are
dated amendments at the bottom; the text above them is not changed.

This repeats docs/protocols/sweep_turns_n10.md on new training data. That sweep (D35, D36) found a
bump share of 0.299 (real) vs 0.239 (rewired), p 0.030 on all 20 runs, but the seed 0-2 lead did not
replicate on seeds 3-9 (p 0.37), so the earlier result is a lead, not a finding. None of the runs
below exist yet, so every seed here is fresh.

## Data and training

- Training set: `data/cache_dart_agg2`, 1,600 episodes: 1,000 expert episodes with DART noise
  (`data/expert_dart`, seeds 0-999) and two DAgger rounds of 300 episodes each
  (`data/dagger_dart/r1`, `r2`), flown by the real-mask / learned-readout / seed-0 reference policy
  (round 1 by the policy trained on the expert episodes alone, round 2 by the one trained on
  expert + r1) and labelled with the plain expert (wander and maneuvers off). Round 2 did not
  improve closed-loop success beyond its interval (75.0% -> 76.0%, n = 200 each), so there is no
  round 3.
- Test set: `data/cache_dart_test`, 100 expert episodes with the same noise, seeds 2000-2099, used
  by no run for training or checkpoint selection.
- Config: `configs/default.yaml` plus `connectome.mask_type`, `heads.decoder` and `seed`, nothing
  else. 50 epochs, per-epoch validation on 10% of the episodes (chosen by the run's seed), and every
  run is scored on `model_best.pt` (best validation heading R^2), as in the earlier sweeps.
- A run that crashes or diverges is rerun with the same seed. Seeds are never swapped or dropped,
  and a low-R^2 run stays in the analysis.

## Conditions

- Mask: `real` vs `rewire`.
- Anatomical readout (`heads.decoder=anatomical`): seeds 0-9 per mask, 20 runs. This is the test.
- Learned readout (`heads.decoder=learned`): seeds 0-2 per mask, 6 runs. Reported per seed; with
  three seeds a permutation test cannot reach p below 0.1, so it is shown but not read as a test.
- Training order: learned s0-2, anatomical s0-2, s3-5, then s6-9. If the clock forces a cut, seeds 6-9
  are dropped for both masks together, decided by elapsed time and before looking at any
  s3-5 or later result. It would be stated in the report.

## Primary metric and test

Gate 3 bump share (`scripts/gate3_heading_decode.py` on `data/cache_dart_test`, anatomical layout,
50-step washout, `ring.anatomical.fraction_mean`), one number per run, definition and 0.5 threshold
unchanged from the earlier protocol. Statistic: mean(real) - mean(rewire) over the seeds of the
anatomical readout. Test: exact two-sided permutation over every relabeling of the runs
(C(20,10) = 184,756 for n = 10 per mask); p is the fraction with |difference| >= the observed one.
`scripts/sweep_stats.py --fresh-from 0`.

## Secondary metrics

Same statistic and test, no correction for multiplicity, descriptive only:

- Heading R^2 and angular error of the decoder on the test set (Gate 3 `decode`), and open-loop
  action MSE on it (`eval_open_loop.py --split all`).
- Ring alignment (Gate 3 `ring.anatomical.alignment`) and the resting-rate-subtracted bump share.
- Topography alignment of the activity-fitted preferred heading with the EB angle
  (`eval_topography.py`, validation episodes of the training cache).
- For the learned readout the same table without a test.

## Closed loop

Seeds 0-2 of all four conditions (12 runs), `model_best.pt`, 200 held-out episodes each (seeds
10000-10199, `scripts/failure_breakdown.py fly`), the same seeds as the reference policies. Success
with Wilson 95% intervals per run and pooled per condition, how the episodes ended (obstacle, side
wall, floor, ceiling, timeout), the drift rate, and where in the corridor. The question is whether the
anatomical-readout runs still fly worse than the learned-readout runs of the same mask. Descriptive: the
pooled intervals treat episodes as independent and ignore the spread between seeds, so the per-seed
values are given next to them.

## Reporting

Every per-seed value of every metric, with per-condition mean, sd (n - 1), min and max, and the
permutation p. Results go to `runs/sweep_dart/stats.json` (`stats_learned.json` for the learned
readout), Gate 3 reports to `runs/gate3_dart/<run>/report.json`.

## Amendment 1 (2026-10-19 12:12 +0530): seeds 6-9 dropped

Written before any anatomical seed 3-5 run exists and before the metrics of the finished runs were
tabulated (real anatomical s0-2, rewired anatomical s0, the six learned runs; only real
learned s0's Gate 3 report was looked at, as a sanity check).

What happened: the first training attempts ran out of GPU memory while another job shared the card,
and an extra raise of the parallel-job limit started a fourth training that crashed at its first
validation, and every queued run behind it crashed the same way within eight minutes. 16 runs are
therefore untrained (rewired anatomical s1-s9, real anatomical s3-s9). Training is deterministic
(real learned s0 reproduced the reference policy's checkpoint bit for bit), so the crashed runs are
simply trained again with the same seed, at most three at a time.

Time spent so far, including an interruption, is already past the budget, so the rule above is
applied: seeds 6-9 are dropped for both masks. The test is on seeds 0-5, six runs per mask
(C(12,6) = 924 relabelings), same statistic and metrics. Order from here: rewired anatomical s1-s2,
real s3-s5, rewired s3-s5. Everything else is unchanged.

## Execution note (2026-10-19 13:23 +0530, not an amendment)

Runs: `xargs -P 3 runs/sweep_dart/train_one.sh` over runs/sweep_dart/jobs.txt (first attempt) and
jobs_rest.txt (after the crash cascade of Amendment 1), each run `python -m flybywire.train.bc
--cache data/cache_dart_agg2 --out runs/sweep_dart/<run> connectome.mask_type=... heads.decoder=...
seed=...`; the scoring is runs/sweep_dart/post.sh on `model_best.pt` (open-loop on the test cache,
eval_topography.py, gate3_heading_decode.py into runs/gate3_dart/<run>, and for seeds 0-2 the
closed-loop flights). Scoring steps that hit a CUDA OOM while three trainings shared the card were
repeated; the open-loop evaluations of rewired learned s0 and rewired anatomical s2 gave up after six
tries and were run again by hand once the card was free, same command. Real learned s0 is
bit-identical to runs/dagger_dart/agg2_real_learned_s0 (same checkpoint file); its closed-loop flights
were repeated and gave the same 76.0% with the same end reason in all 200 episodes (positions differ in
some episodes: GPU nondeterminism). Not in the protocol: closed-loop flights of real anatomical s3-5, so
that every real-mask anatomical run has one (the demo candidate is picked among them).
`scripts/sweep_stats.py --root runs/sweep_dart --gate3 runs/gate3_dart --fresh-from 100` (and
`--decoder learned`) writes stats.json / stats_learned.json; the closed-loop tables are
closed_loop_report.json (seeds 0-2, four conditions) and closed_loop_report_real_anatomical_s3_5.json.
