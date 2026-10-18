# Protocol: real vs rewired wiring at n = 10 seeds

Written 2026-10-18 23:29 +0530, before any run of seeds 3-9 was started (the first started 23:30). Seeds 0-2 were trained
and looked at earlier (D35): that is where the lead (bump share 0.34 +/- 0.10 real vs 0.19 +/- 0.02
rewired) came from. This protocol is the test of that lead. Edits after this point are dated
amendments at the bottom; the text above them is not changed.

## Conditions

- Readout: `heads.decoder=anatomical` for every run.
- Mask: `connectome.mask_type=real` vs `connectome.mask_type=rewire`.
- Seeds 0-9 per condition, 20 runs. Seeds 0-2 exist (runs/sweep_turns/{real,rewire}_anatomical_s{0,1,2}).
  Seeds 3-9 are trained with the same config (configs/default.yaml with only the two overrides
  above and `seed`), the same training data (`data/cache_turns`), the same sweep tool
  (`python -m flybywire.train.sweep --resume`), and are scored on `model_best.pt`.
- A run that crashes or diverges is rerun with the same seed. Seeds are never swapped or dropped,
  and a low-R^2 run stays in the analysis.
- The learned-readout runs (seeds 0-2) are not part of this test.

## Primary metric

Gate 3 bump share: `scripts/gate3_heading_decode.py` on `data/cache_turns_test` (100 episodes,
never trained on), anatomical layout, 50-step washout, `ring.anatomical.fraction_mean` in
`report.json`. One number per run. The 0.5 threshold is unchanged.

## Test statistic and test

Statistic: mean(real) - mean(rewire) of the primary metric over the 10 seeds of each condition.
Test: exact two-sided permutation test over all C(20,10) = 184,756 ways to relabel the 20 runs as
10 real and 10 rewired; p is the fraction of relabelings with |difference| >= the observed one
(the observed labeling is one of them). Enumerating them all is cheap, so no Monte Carlo.

Because seeds 0-2 produced the hypothesis, the same test is also reported on seeds 3-9 alone
(7 vs 7, C(14,7) = 3,432 relabelings). That subset is the only part of the data the hypothesis was
not chosen on.

## Secondary metrics (same statistic and test, no correction for multiplicity, descriptive)

- Heading R^2 and angular error of the decoder on `data/cache_turns_test` (Gate 3 `decode`).
- Ring alignment: bump angle vs heading, anatomical layout (Gate 3 `ring.anatomical.alignment`).
- Topography alignment of each EPG neuron's activity-fitted preferred heading with its EB angle
  (`eval_topography.py`, `reliable_anatomy.activity_phi.alignment`). The decoder phi is fixed by the
  EB angle under this readout and is not scored.
- Any corrected sharpness measure from the synthetic check of the share definition (see
  amendments), if one is needed. It would be reported next to the primary, same 0.5 threshold.

## Reporting

Every per-seed value of every metric is reported, with per-condition mean, sd (n - 1), min and max,
and the permutation p. Results go in `runs/sweep_turns/stats.json`.

## Amendment 1 (2026-10-18 23:39 +0530): share definition, synthetic check, resting-rate sensitivity

Written before any seed 3-9 run had finished or been evaluated. Training had been started at 23:30
in parallel with this check; the check uses only synthetic profiles and the six seed 0-2 runs.
The header time above was corrected from 23:33 to 23:29 (file mtime); nothing else above was edited.

Definition of the primary metric, as coded in `flybywire.eval.ring.bump` (key `fraction`): per
frame, the wedge value is the mean rate of the reliable EPG neurons (47 of 51) whose EB angle
falls in that 22.5 deg wedge (2-4 neurons each); P = rfft of the 16 values; share =
|P_1|^2 / sum over k = 1..8 of |P_k|^2. The mean (k = 0) is not in the denominator, so the Nyquist
harmonic counts once and a floor added equally to every wedge cannot change the share. A run's
value is the mean over all frames after the first 50 core steps of each of the 100 test episodes
(95,000 frames). Pure cosine 1, white noise 0.126.

Synthetic check (`scripts/bump_sharpness_check.py`, `runs/bump_check/report.json`):

- A uniform floor, softplus(0) = 0.69 or 5, changes the share by < 1e-15.
- The trained cores do not sit at softplus(0): 5th-percentile EPG rates are 0.03-0.06 on average
  per run (largest single neuron 0.4-1.4), typical bump height (frame peak wedge minus median
  wedge) 3.4-5.0.
- A Gaussian bump of FWHM 53 deg scores 0.49-0.50 and 0.5 is reached at FWHM 52.7 deg with no
  floor and 52.7-52.9 deg with each run's measured per-neuron floor and bump height; wide bumps
  reach 1.0. The threshold is reachable and the floor does not move it.
- Shares map to ideal-bump widths (Gaussian, real layout, mean over bump positions): 0.19 is
  about 15 deg (a bump inside one wedge, anywhere in 10-25 deg depending on position, and close
  to the 0.126 noise level), 0.34 about 35 deg, 0.50 about 53 deg, 0.80 about 94 deg.

Decision: no corrected sharpness measure is adopted, the primary metric and the 0.5 threshold are
unchanged. As a sensitivity check one more secondary value is reported with the same statistic
and test: the share after each neuron's resting rate (its 5th percentile over the scored frames
of that run) is subtracted before the wedge means, `ring.anatomical.fraction_mean_floor_subtracted`
in the Gate 3 report. The quantile is fixed here. Seeds 0-2 give 0.257, 0.290, 0.441 (real) and
0.170, 0.179, 0.200 (rewired); it never differs from the primary by more than 0.01. If the
real-vs-rewired conclusion differed between the two versions, that will be stated.

## Execution note (2026-10-19 02:14 +0530, not an amendment)

Seeds 3-9 for both masks trained 23:30-00:31 with `python -m flybywire.train.sweep --resume
--cache data/cache_turns --out runs/sweep_turns sweep.decoders=[anatomical]
sweep.seeds=[0..9] sweep.max_parallel=3`; all 14 exited 0 and each config.yaml equals its seed 0-2
sibling's apart from `seed`. Each run was scored once with `eval_topography.py` and
`gate3_heading_decode.py` on `model_best.pt`. The first evaluation attempts of real s4 and s5 ran
out of GPU memory while training was still using the card; their partial logs were removed and
both were evaluated again after training ended. Gate 3 was rerun on seeds 0-2 to add the
resting-rate field; their primary values are bit-identical to the earlier reports.
`scripts/sweep_stats.py` produces `stats.json`.
