# Decisions log

Append-only. Each entry: date, decision, rationale. Deviations from the setup spec are marked
**[deviation]**.

---

## 2026-10-16 — D1. A single recurrent population, not stacked residual blocks

The central complex core is one population of N rate neurons whose only neuron-to-neuron path is
`r @ (W * M)`. No residual stream, no LayerNorm, no attention, no dense layer over the neuron axis.

*Rationale.* A residual stream would carry EPG activity around the mask, so silencing EPG would not
actually silence its downstream influence and lesion experiments would be uninterpretable.
LayerNorm across neurons is itself an all-to-all interaction that bypasses the connectome. With a
single masked population, "which neurons can influence which" is exactly the FlyWire mask.

## 2026-10-16 — D2. Directed, degree-preserving rewire; `networkx.double_edge_swap` rejected

The null model uses repeated directed two-edge swaps (`a→b, c→d` ⇒ `a→d, c→b`, four distinct
endpoints, reject if a new edge exists or is a self-loop), 10 × |E| successful swaps, seeded.

*Rationale.* This preserves every node's in-degree and out-degree exactly, so the null differs from
the real connectome only in *which* partners are connected. `networkx.double_edge_swap` works on
undirected graphs only and would destroy the in/out-degree split that defines e.g. ER as pure
input and PFL3 as mostly output neurons. An Erdős–Rényi (`er`) mask with the same N and edge count
is kept as a weaker second null.

## 2026-10-16 — D3. Body-frame velocity + yaw-rate action space

Policy outputs `[-1, 1]^4` → `[v_forward, v_lateral, v_vertical, yaw_rate]` (body frame), tracked by
gym-pybullet-drones' PID controller (target yaw integrated from `yaw_rate`, body velocity rotated
into a world-frame target velocity).

*Rationale.* Raw thrust or body-rate control would require writing an inner-loop attitude
controller and would make both the scripted expert and BC fragile. Yaw rate remains the steering
variable that PFL3 controls, which is the biological claim of interest.

## 2026-10-16 — D4. Heading relative to the episode start; whole-episode BC

The heading target is `(sin, cos)(wrap(yaw_t − yaw_0))`. BC sequences always start at step 0 of an
episode; truncated BPTT carries state across 128-step windows and detaches at window boundaries.

*Rationale.* The network has no compass input; an absolute heading is not observable from optic
flow alone. Heading relative to start *is* observable by integrating yaw rate, but only if the core
integrates from t = 0 — random mid-episode chunks would give the core an unknown offset.

## 2026-10-16 — D5. LPTC `gains` mode is the default

LPTC weights are fixed ego-motion templates (Longuet-Higgins & Prazdny flow fields projected onto
calibrated T4/T5 preferred directions); only a per-(channel, subtype) gain (8 × 8 = 64 params) is
trained by default. `frozen` (0 params) and `full` (~46k params) are available.

*Rationale.* Keeps the connectome-masked core as the dominant trainable component, so differences
between mask conditions are not absorbed by a large learnable visual front end.
`scripts/count_params.py` reports the ratio.

## 2026-10-16 — D6. fp32 only

No AMP, no autocast, no bf16, no `torch.compile` (for now). A test asserts every parameter and buffer
is fp32. (The on-disk T4/T5 cache and stored hexals are fp16 for size; they are upcast to fp32 on
load before entering the model.)

*Rationale.* The ring-attractor dynamics are marginally stable; low-precision rounding produces slow
drift of the bump that looks biological but is a numerical artefact.

## 2026-10-16 — D7. The controlled comparison runs in the BC regime only

Real vs. rewired (and optionally ER) masks are compared with 3 seeds each under behavior cloning.
PPO fine-tuning is run on the real connectome for the demo.

*Rationale.* BC is a supervised problem with low variance across seeds, so mask effects are
measurable with 3 seeds; PPO variance would swamp them at this budget. One cache serves every
condition and seed.

---

## Installation and environment

## 2026-10-16 — D8. Python 3.12 instead of 3.10 **[deviation]**

The spec suggests Python 3.10 "to match gym-pybullet-drones' documented setup". The current
`main` branch of gym-pybullet-drones (`pyproject.toml`, version 2.2.0) declares `python = "^3.12"`,
and pip enforces `Requires-Python` even with `--no-deps`. flyvis 1.2.0 declares
`>=3.9,<3.13`. The only Python satisfying both is **3.12**, so `environment.yml` pins
`python=3.12`.

## 2026-10-16 — D9. PyTorch from the CUDA 13.0 index (cu130)

PyTorch's release matrix lists 2.14 with stable CUDA 12.6 / 13.0 / 13.2; the cu128 index tops out at
torch 2.11. gym-pybullet-drones declares `torch = "^2.13"`. The spec allows "a newer CUDA index if
one is the current stable", so `setup_env.sh` uses `https://download.pytorch.org/whl/cu130`
(override with `TORCH_INDEX=...`). The RTX 5060 Ti (sm_120) needs CUDA ≥ 12.8; cu130 satisfies this.
`check_env.py` verifies `sm_120` is in `torch.cuda.get_arch_list()`.

## 2026-10-16 — D10. Installation outcome (Phase 2)

- `scripts/setup_env.sh` ran cleanly end to end. Installed: Python 3.12.15, torch 2.14.1+cu130
  (CUDA 13.0, cuDNN 9.24), flyvis 1.2.0, gymnasium 1.4.0, pybullet 3.2.7, gym-pybullet-drones 2.2.0
  (commit `01563120`), numpy 2.5.2, numba 0.68.0. `pip check`: no broken requirements.
- `pip install flyvis` did **not** replace torch (verified by the script).
- gym-pybullet-drones: a `pip install --dry-run --report` showed that installing it *with*
  dependencies would replace no already-installed package, so it was installed normally
  (not `--no-deps`). This pulls in stable-baselines3 2.9.0 as a transitive dependency; the project
  does not use SB3 (PPO is written by hand, per spec).
- pybullet 3.2.7 has no prebuilt cp312 wheel; pip built it from source (~3 min, needs a C++
  toolchain, which the host had). EGL hardware rendering works: 1.7 ms/frame vs 3.9 ms/frame for
  TinyRenderer at 160×160 on the RTX 5060 Ti.
- `flyvis download-pretrained --skip_large_files` downloaded ~3.5 MB of zips; `du -sh data/flyvis`
  = **10 MB** (50 `flow/0000/*` models). The models are small because flyvis has only a few hundred
  free parameters per network. On the first network initialisation flyvis additionally builds a
  101 MB connectome cache in `data/flyvis/connectome/`, so the steady-state footprint is 111 MB.

## 2026-10-16 — D11. flyvis resolves `FLYVIS_ROOT_DIR` relative to the CWD

`flyvis/__init__.py` calls `dotenv.load_dotenv(find_dotenv(usecwd=True))` and then
`Path(os.getenv("FLYVIS_ROOT_DIR")).absolute()` — i.e. `./data/flyvis` from `.env` is resolved
against the *current working directory*, and the directory is silently `mkdir`ed if missing. Running
from a subdirectory would therefore look for models in the wrong place. `flybywire.paths.setup_flyvis_env()`
sets `FLYVIS_ROOT_DIR` to an absolute path (repo-relative) **before** flyvis is imported; all
flybywire code imports flyvis only through `flybywire.paths.import_flyvis()`, which calls it first.

---

## Implementation (Phase 3)

## 2026-10-16 — D12. `import flyvis` sets torch's global default device to CUDA

`flyvis/__init__.py` runs `torch.set_default_device(device)` at import. Left alone, every later
`torch.zeros(...)` in our code would silently land on the GPU. flyvis internals rely on the default
device (e.g. `Network.target_sum` allocates with `torch.zeros(...)` and no device), so we cannot just
reset it globally and call flyvis. `flybywire.paths.import_flyvis()` restores the previous default
right after import, and every flyvis call (network construction, BoxEye, stepping) runs inside
`with torch.device(dev):`. All flyvis imports go through that helper.

## 2026-10-16 — D13. Frame-by-frame flyvis stepping API

Verified in `flyvis/network/network.py` (1.2.0): `Network.forward(x, dt, state, as_states=True)`
returns per-step states; passing the last one back as `state=` carries state across calls (the
`public` stepper mode). `forward` re-runs `clamp()` and `_param_api()` on every call; for a frozen
net the parameter namespace is constant, so the default `fast` mode computes it once and calls the
private `Network._next_state(params, state, x_t, dt)` directly (the exact arithmetic `forward`
loops over). The `PPNeuronIGRSynapses` state is only `nodes.activity` (B, 45,669); edges carry no
state, which the stepper asserts. Both modes are checked against a single full-sequence call by
Gate 1. *Risk:* `_next_state`/`_state_api` are private; a flyvis upgrade could break `fast` mode —
`public` mode is the fallback and Gate 1 re-validates both.

Also: `NetworkView.init_network()` leaves some parameters with `requires_grad=True` (eval mode
alone does not freeze them); the stepper sets `requires_grad_(False)` on all of them. The
pretrained models were trained at dt = 0.02 s; we integrate at dt = 0.01 s (spec), which flyvis
accepts (it warns only for dt > 1/50). All 8 T4/T5 types have 721 cells in exactly the
`get_hex_coords(15)` order, which is also BoxEye's output order.

## 2026-10-16 — D14. LPTC input = rectified T4/T5 activity minus grey baseline

flyvis `activity` is membrane voltage; synaptic output in the model is `relu(activity)`.
`LPTCPool` uses `T_d(p) = relu(activity) − baseline_d`, where `baseline_d` is the mean
`relu(activity)` of subtype d at the grey steady state (written to `direction_map.json` by the
calibration). The cache stores raw voltage (lossless w.r.t. this choice), so this can be revisited
without rebuilding the cache.

## 2026-10-16 — D15. Flow-template conventions and `env.camera_tilt_deg` **[config addition]**

Channels are defined in the body frame (FLU), right-handed: roll/pitch/yaw about +x/+y/+z
(positive yaw = turning left), forward/lateral/vertical along +x/+y/+z. They are rotated into the
camera frame (X right, Y down, Z forward), which is pitched down by the new key
`env.camera_tilt_deg` (initial 10°; the spec says "slight downward tilt" without a number, and the
templates must use the same value as the env camera). Hexal image positions come from
`get_hex_coords` mapped through BoxEye's receptor geometry (`row = d(u + v/2)`, `col = d·v`) on the
391×391 grid the frame is resized onto, then to focal units for the 110° pinhole.

## 2026-10-16 — D16. Decision/action alignment (defined once in `flybywire/clocks.py`)

Decision k is read out after core step 4k+3 (the last tick of its 4-frame window); action a_k is
held over physics steps 8(k+1)…8(k+1)+7. The first window (physics 0–7, frames 0–3) is a hover
pre-roll executed by `reset()`, so an episode of K decisions has exactly T = 4K frames/core steps
(matches the spec's "300 trajectories × 1000 steps" for 250 decisions). Consequently the env
returns **4 frames per step** (`(4, H, W)` uint8) and `reset()` returns the 4 pre-roll frames.

## 2026-10-16 — D17. Input-dependent alpha is elementwise

`alpha = sigmoid(a + c ⊙ u_t)` with a per-neuron gain `c` (init 0, flag off by default). A general
`g(u_t)` mapping across neurons would be a non-connectomic cross-neuron path (violating invariant
1); elementwise keeps each neuron's time constant a function of its own input only. Since `u_t` is
nonzero only on ER, only ER time constants are input-dependent.

## 2026-10-16 — D18. Other config additions **[config addition]**

- `connectome.flywire.*`: snapshot, file names and column names for the Codex tables (defaults:
  `connections.csv.gz` with `pre_root_id, post_root_id, syn_count`; `consolidated_cell_types.csv.gz`
  with `root_id, primary_type`; optional `names.csv.gz` with `root_id, name`). These are best
  guesses for snapshot 783 and **must be verified** against the downloaded files; the loader
  prints the columns it found and exits on a mismatch.
- `lptc.direction_map`: path to the direction map (null → `$FLYVIS_ROOT_DIR/direction_map.json`).
  The smoke run uses it to point at a provisional map when calibration has not been run yet.

## 2026-10-16 — D19. Anatomical ring layout: PB glomerulus → wedge convention (approximate)

Instance names are parsed for a PB glomerulus `L1…L9` / `R1…R9`; glomeruli 1–8 map to 16 wedges
as `wedge = 2(g − 1) + (0 if R else 1)` (each PB half tiles the ring, halves interleaved). This is
an approximation of the EB↔PB mapping in Hulse et al. 2021, glomerulus 9 is unmapped, and the
real FlyWire names may not contain glomeruli at all; `auto` then falls back to `tuning`. The figure
always states which layout was used.

## 2026-10-16 — D20. PPO will use gymnasium `AutoresetMode.SAME_STEP`

gymnasium ≥ 1.0 defaults vector envs to `NEXT_STEP` autoreset (the step after termination ignores
its action and returns the reset obs). For a recurrent policy with per-env core/flyvis state,
`SAME_STEP` (reset inside the terminating step; final obs in `info["final_obs"]`) avoids a wasted
dummy step and makes "zero env i's state, then consume the new episode's frames" explicit.
Verified in the installed `gymnasium/vector/async_vector_env.py` (1.4.0).

## 2026-10-16 — D21. Null-model seed = training seed; TBPTT updates per window

- `build_connectome(cfg)` seeds the rewire / ER null with `cfg.seed`, so the 3 rewired seeds are 3
  different null graphs (variance over null instances, not just over initialisations).
- BC takes one optimizer step per 128-step TBPTT window (standard truncated BPTT), carrying the
  detached state into the next window of the same episode batch.

## 2026-10-16 — D22. Small additions to the module layout

`flybywire/paths.py` (safe flyvis import, D11/D12) and `flybywire/vision/stimuli.py` (drifting
panoramas and gratings shared by the smoke run, Gate 1 and calibration). `measure_directions()`
lives in `flybywire/vision/directions.py` so the calibration script and the smoke run share it.

---

## Gate 1 and calibration (Phase 5)

## 2026-10-16 — D23. Direction calibration uses 12 directions, not 4 **[deviation]**

The spec's protocol (gratings moving +x, −x, +y, −y; preferred vector `(R+x − R−x, R+y − R−y)`) was
implemented first. With 24/40 px periods at 2 Hz it gave near-zero direction selectivity for T4d
(DSI 0.03) and T5b (0.035), so their "preferred directions" were noise. A wider probe (12
directions × 6 period/TF conditions, central hexals) showed: T5b is well tuned only at larger
periods / higher TF; T4d is weakly tuned under every condition. The calibration now uses 12
directions every 30° (a superset that still contains ±x, ±y, whose responses are still recorded),
conditions (40, 64 px) × (2, 4 Hz), hexals with normalised radius < 0.6 (avoids border effects),
and a vector-sum estimate. `direction_map.json` stores the full tuning curves.

Result for `flow/0000/000` (image coords, x right, y down):

| subtype | angle | DSI |
|---|---|---|
| T4a | −177.4° | 0.41 |
| T4b | +8.2° | 0.52 |
| T4c | −61.3° | 0.51 |
| T4d | −76.4° | 0.23 |
| T5a | −177.0° | 0.48 |
| T5b | +21.6° | 0.33 |
| T5c | −117.6° | 0.77 |
| T5d | +120.3° | 0.54 |

Note: the a/b pairs are horizontal and opposite. The T5 c/d pair is opposite and vertical at ±118°
(not ±90°; consistent with BoxEye's skewed hex lattice: `y = d(u + v/2)`, `x = d·v`). **T4d is
weakly selective and points upward next to T4c rather than opposite it** — a property of this
pretrained model under these stimuli, not corrected here. In `gains`/`full` LPTC modes the
per-(channel, subtype) gains can down-weight it.

## 2026-10-16 — D24. Gate 1 result: PASS

`runs/gate1/report.json`: frame-by-frame stepping with carried state matches a single
full-sequence `Network.forward` call to max |Δ| = 4.8e-7 (relative 2.7e-7; scale 1.79) in both
`fast` and `public` modes — the non-zero residual is consistent with non-deterministic
`scatter_add` accumulation order on the GPU. Throughput (`fast`): B = 1 → 5,640 steps/s;
B = 12 → 634 batched steps/s = 7,605 env-steps/s (`public`: 2,655 / 568). Criterion (≥ 100
batched steps/s at B = 12) met with margin; `fast` is the default stepper mode.

Observation for later: flyvis's BoxEye is now the slower stage at B = 12 (419 calls/s, ~2.4 ms):
it resizes 160² → 391² and convolves samples one at a time. Replacing it with an equivalent batched
`avg_pool2d` is a cheap optimisation if PPO rollouts become BoxEye-bound (must be checked for
exact equivalence).

---

## Documentation (Phase 6)

## 2026-10-16 — D25. FlyWire thresholding order

The spec says: sum synapse counts per (pre, post) pair across neuropils, then drop pairs below
`min_synapses` (5). If the downloaded connections table is already thresholded per
(pre, post, neuropil) row, pairs whose synapses are spread over several neuropils are undercounted
before our sum. The README therefore recommends an unthresholded variant of the connections table
when Codex offers one. The loader cannot detect which variant it was given; whoever downloads the
data should note which file was used here.

## 2026-10-17 — D26. Comment and docstring cleanup; `paper/` removed

No-behavior-change pass over `flybywire/`, `tests/` and `scripts/` to match the code-style section in
AGENTS.md: dropped banner comments, comments that restated the next line, dead `noqa`s and
`(STUB)` docstrings that repeated the function name. Two small code simplifications, both
behavior-neutral: `python-dotenv` is a hard requirement, so `config.py` and `paths.py` import it at
module level instead of behind an `ImportError` guard, and redundant `Path(data_dir(cfg))` /
`Path(runs_dir(cfg))` wraps are gone (both already return `Path`). Module docstrings that argparse
prints as `--help` text (`build_cache`, `calibrate_t4t5_directions`, `check_env`, `train.sweep`)
were left as they were.

This is a project, not a paper, so `paper/` (writeup skeleton and `references.md`) was deleted. The
citations that only lived in `references.md` are now in the README.

## 2026-10-17 — D27. Toy connectome removed; tests run on the FlyWire graph (N = 457)

`connectome.source` is `flywire` and `flybywire/connectome/toy.py` is gone. Snapshot 783,
`min_synapses` 5: N = 457, 17,041 edges (ER 278, EPG 51, PEN 42, PEG 20, D7 42, PFL3 24). Tests and
`make smoke` load `data/flywire/processed/cx_graph.npz`. The `conn` fixture skips with a reason when
the file is missing, so a clone without the FlyWire download runs 42 of 59 tests. `make params`:
17,041 on-mask `W`, 20,729 parameters trained in BC, 54,266 with the critic (`lptc.mode: gains`).

Things the toy graph hid:
- ER receives 7,804 edges and PFL3 sends 40, so ER is not pure input and PFL3 not pure output. The
  toy-only assertions on that are gone; invariants 1-10 are unchanged and pass.
- Instance names look like `EB.27`, `PB.EB.2`, `FB.LAL.17` (neuropil + index), so no PB glomerulus
  parses (0 of 51 EPG). `heads.ring_layout: auto` therefore always resolves to `tuning` here (see
  D19). The anatomical layout is tested on synthetic `EPG_R3`-style names only.
- Directed rewire at N = 457 completes all 170,410 target swaps in 0.2 s.

## 2026-10-17 — D28. FlyWire skeletons downloaded for the 3D view

`data/flywire/skeletons/sk_lod1_783_healed_ds2.parquet` (5,355,543,468 bytes, md5
`a4c104776f33ec539ef859064c4de3df`, matches Zenodo) from Zenodo record 10877326
(Schlegel et al. 2024 supplement, CC BY 4.0). Skeletons for all 139,273 neurons of release 783,
generated from lod 1 meshes and 2x downsampled. Columns: `node_id`, `parent_id`, `radius`, `x`, `y`,
`z` (nm), `neuron` (root id); per-neuron soma node ids are in the parquet schema metadata.

All 457 neurons in `cx_graph.npz` are present (1,137,123 nodes in total, 1,068 to 6,895 per
neuron), so only a small subset needs reading. Not used by any code yet.

---

## Environment and expert data

## 2026-10-18 — D29. Corridor env, scripted expert and vec env

**Env.** `BaseAviary` (CF2X, `Physics.PYB`) with `pyb_freq = ctrl_freq = 200`, so one aviary step
is one physics step, and `DSLPIDControl` every physics step. Velocity tracking as in the library's
`VelocityAviary` (`target_pos` = current position, command in `target_vel`). Target yaw integrates
`yaw_rate`, which is also passed as the yaw entry of `target_rpy_rates`. Body velocity is rotated
into the world by the current yaw. A 1 rad/s yaw-rate command is tracked within 0.1 rad/s.

Scene: corridor along world +x (x 0-30 m, y +/-1.5 m, z 0-2.5 m) with walls, ceiling and end walls;
the floor is pybullet's `plane.urdf` checker. Eight floor-to-ceiling obstacles, cylinders
(r 0.15-0.3 m) or boxes (half sizes 0.15-0.35 m), evenly spaced from x = 4 to 28 m with +/-0.5 m
jitter, centres within 0.7 m of the axis. Start at x = 1 m, y +/-0.3 m, z 1.0-1.4 m, yaw 0, so
`yaw_0` is the corridor axis. Any contact point (floor included) is a collision. x >= 29 m or
leaving the box also ends the episode. The progress reward is in m/s (dx * control_hz): as
distance per decision (~0.05 m) the heading term would dominate and hovering would pay.

**TinyRenderer by default (`env.renderer: tiny`) [deviation].** The spec says EGL first. In DIRECT
mode the EGL plugin always draws shadow maps; `shadow=0`, `COV_ENABLE_SHADOWS` and
`shadowMapIntensity` change nothing (identical pixels). In a closed corridor that means shadow
acne (moire) on walls lit at grazing angles and big shadow-boundary patches, and both move with
the camera: image motion that isn't ego-motion. TinyRenderer with `shadow=0` is clean but costs
8.6 ms per frame vs 1.7 ms, ~35 ms per decision, ~9 s per episode. `auto` still tries EGL first.
Every body also gets an explicit texture, since OpenGL otherwise puts a fine checker on
primitives that aliases at a distance.

**Expert** (new `expert:` config section **[config addition]**). Potential field: pull toward a
point 3 m down the centre line, or, when the nearest obstacle within 3 m blocks the centre (half
width + 0.45 m), toward the middle of the wider gap beside it at the obstacle's x; push from
obstacle and wall surfaces within 0.6 m. Field direction plus a seeded OU wander (sd 8 deg,
tau 1 s) gives the desired heading (max 45 deg), tracked by a heading PID (2.5, 0.2, 0.15). Speed
1.5 m/s along that heading, down to 40% near obstacles; altitude held. Without the wander every
dodge looked the same. Seeds 0-299, dynamics only: 300/300 full episodes, no collisions, mean
progress 11.9 m (about 4 of the 8 obstacles), yaw-rate sd 0.36 rad/s, mean peak |rel_yaw| 32 deg.
No episode reaches the corridor end in 10 s at this speed.

**Trajectories.** Window k (window 0 = reset pre-roll) is decision k's input. The window returned
by the final step has no decision and is dropped, so T = 4K in every file.

**Vec env.** `AsyncVectorEnv(shared_memory=True, context="spawn", autoreset_mode=SAME_STEP)`.
Spawned so workers don't inherit the main process's CUDA context.

## 2026-10-18 — D30. Expert data, T4/T5 cache, Gate 2: PASS

`make collect`: seeds 0-299, `wall_texture: none`, TinyRenderer, 12 workers, 422 s. All 300
episodes ran 250 decisions (T = 1000) without a collision; mean progress 11.9 m. 127 MB of
trajectories (the README guessed ~0.5 GB; flat walls compress well) and 18 MB of raw frames for
episodes 0-19. `make cache`: 300,000 x 8 x 721 fp16, 3.46 GB, 53 s.

Gate 2 (`runs/gate2/report.json`): ridge from all 5,768 T4/T5 voltages at frame t to body yaw rate
at frame t. Split by trajectory 210 / 30 / 60 (train / validation / test, seed 0), first 50 frames
per episode dropped (onset transient from the grey steady state), penalty picked on validation
(3e-3 per sample, inside the grid), refit on train + validation. **Test R^2 = 0.787** on 57,000
held-out frames (train + validation 0.823). Pooled 4 x 4 grid per subtype (128 features): 0.741.
Passes with untextured walls, so the checker / noise fallback isn't needed.

Not part of the gate: shifting the target against the features (pooled, same split) gives R^2
0.68 at -40 ms, 0.75 at 0, 0.84 at +40 ms, a peak of 0.89 at +80 ms and 0.88 at +120 ms. T4/T5
activity reflects the yaw rate of ~80 ms earlier, which is the right sign for a sensory delay (a
flipped alignment would peak at negative lags). The core will get yaw rate ~80 ms late.

## 2026-10-18 — D31. First BC run on the real connectome (seed 0)

`python -m flybywire.train.bc --out runs/bc/real_s0`, default config: 270 train / 30 validation
episodes, 50 epochs, 3,600 updates, 131 s. Validation: heading R^2 0.636 (joint sin/cos), mean
heading error 6.5 deg (predicting 0 gives 11.1 deg), action MSE 0.024 (predicting the mean gives
0.039). Per action: yaw rate R^2 0.47, lateral 0.47, forward 0.28; vertical has almost no
variance because the expert holds altitude.

Heading error grows through the episode, 3.7 deg in the first second to 8.5 deg in the last, so
the core integrates with drift. On training episodes heading R^2 is 0.80 (error 5.3 deg) with
the same action MSE, so the heading readout partly overfits 270 episodes. That is below the
Gate 3 threshold of 0.8, though Gate 3 itself (with the ring check) isn't implemented yet. Loss
was still falling at epoch 50; bc.py has no per-epoch validation, so the best stopping point is
unknown. Headings are small here (rel_yaw sd 14.5 deg), which makes R^2 sensitive to a few
degrees of error.

## 2026-10-18 — D32. EPG anatomical angles, topography baseline, closed-loop eval

`scripts/epg_anatomy.py` gives each EPG neuron an angle around the EB from the FlyWire skeletons.
The EB is the blob of smoothed ER node density above 30% of peak that contains the joint ER/EPG
peak, which leaves out the bulbs. Ring normal = smallest PCA axis of EPG nodes in it; zero is
dorsal, positive toward +x (in-plane PCA axes were unstable on a nearly round ring). 47 of 51
EPGs get stable angles (1-2 deg RMS across thresholds); the 4 EPGt barely enter the EB and are
flagged. Angles cover all 16 wedges (2-4 each), and angular distance predicts PEN/PEG/D7 partner
overlap (Spearman -0.76, p < 1e-4), so the angles are real anatomy, not fitting noise.

Topography on bc/real_s0 is null: alignment of preferred heading with EB angle 0.07 (rates) and
0.16 (decoder weights), p >= 0.5. Not conclusive: headings are small (sd 14.8 deg), so the cos h
term is poorly identified (SE 5x that of sin h); EPG rates carry a push-pull 1-D heading signal
unrelated to anatomy. A perfect ring would still score 0.96 given the bootstrap noise in phi.
Rerun on the large-turn data.

Closed loop (`scripts/eval_closed_loop.py`, held-out seeds 10000-10049, env and expert from the
checkpoint's config): the BC policy succeeds in 62% of episodes (19 obstacle hits), 9.4 m mean
progress, against 100% and 12.1 m for the expert. Heading error 5.9 deg. Action MSE against the
expert's label on states the policy visits is 0.199, eight times the open-loop 0.024, mostly yaw
rate and lateral velocity: the compounding error behavior cloning is known for. `--record` saves
those episodes with expert labels (wander off: in DAgger the policy's own errors diversify the
states, and wander in a label is noise the input can't predict).

## 2026-10-18 — D33. Turning expert, yaw-wrap fix, data/expert_turns

The plain expert barely turns (rel_yaw sd 14.5 deg, peak ~32 deg), so heading decoding is fragile
and the EPG ring never sees more than a wedge or two. New `expert.maneuvers`: at random times, when
the nearest obstacle surface is > 1.2 m and the next obstacle front > 2.2 m ahead, the expert slows
to 0.3 m/s, yaws to +/-(90..360) deg, holds up to 1 s, then returns to heading 0 the short way (270
and 360 end as full rotations). The yaw reference follows a trapezoidal rate profile (2.5 rad/s,
8 rad/s^2) and replaces the heading target, so the 45 deg clip doesn't apply; the world-frame
potential field keeps avoiding obstacles. `action.limits.yaw_rate` goes to +/-3 rad/s so turns
don't saturate; that changes action normalisation for new runs only. Without the key, expert and
env reproduce the old data bit for bit (all 300 stored episodes, and the config in bc/real_s0).

DSLPIDControl's yaw D term differences cur_rpy - last_rpy, which jumps 2 pi when yaw crosses
+/-pi (a 0.18 rad/s kick in one physics step). `_physics_step` now unwraps `last_rpy[2]` next to
the current yaw; a no-op when yaw doesn't cross pi.

Seeds 0-999: rel_yaw sd 56.7 deg, 35% of episodes with a full rotation, 2 collisions (seeds 354,
760; the old expert hits both, plus three more, through its centre-line rule for obstacles that
just miss the axis), mean progress 8.7 m (old 11.8). Requiring 2.0 m ahead at 0.5 m/s added a
collision after a spin ended 0.9 m from a box; 2.5 m cut full rotations to 22% because obstacles
are 3.4 m apart. Collection 23.5 min, 432 MB; cache (999,256 x 8 x 721) 11.5 GB in 156 s. Gate 2 on
it (700/100/200 split): test R^2 0.880 full, 0.778 pooled; best penalty at the grid's lower edge.

## 2026-10-18 — D34. BC on the turning data, DAgger, per-epoch validation

train_bc now scores the validation episodes after every epoch (history gets val_*) and saves the
epoch with the best val heading R^2 as model_best.pt. At lr 1e-3 with no decay val R^2 jumps
between epochs (0.78 to 0.98 within the last ten), so the last-epoch model.pt is unreliable
(closed-loop success 42% vs 58% for model_best.pt). bc/turns_s0 (1000 turning episodes, 11,600
updates, 548 s): heading R^2 0.978 val, 0.975 on 100 fresh expert episodes (data/expert_turns_test,
seeds 2000-2099, used by no run for training), mean error 4.2 deg, action MSE 0.021. Gate 3's
R^2 > 0.8 now holds open-loop for every run; the ring check isn't done. Shifting the target back
in time helps by at most 0.001 (best at 1-2 frames). That doesn't show the 80 ms delay of D30 is
gone: heading moves little in 80 ms, so this R^2 is nearly blind to it. Target is unchanged.

Closed loop on seeds 10000-10199 (n=200): turns_s0 59.5%, expert 100%. DAgger rounds fly the best
checkpoint on 300 fresh seeds (21000.., 22000..), label each visited state with the scripted
expert, and retrain from scratch on the aggregate (build_cache.py takes several --traj-dir).
Labels have wander and maneuvers off: maneuver onsets are random and invisible to the policy, so a
maneuver label is noise there. Flying round 1 both ways: 70.5% vs 64.5% success, heading error 5.4
vs 7.3 deg, val action MSE 0.044 vs 0.060; the success gap is within noise, the rest consistent.
Round 1 70.5%, round 2 57.5% (SE ~3.4 points), so no round 3: single-seed run-to-run variation
looks as large as the DAgger effect. The imitation gap on visited states (plain labels) does drop
from 0.227 to 0.117. Open-loop action MSE is ~10x lower than on visited states; one guess, not
tested, is that labels depend on obstacle positions the untextured scene barely shows.

Topography (turns_s0, dagger1, dagger2): phi is now well determined (bootstrap half-width 4-5 deg,
cos/sin SE ratio 0.8) and spread over the circle, and activity phi matches decoder weights
(0.55-0.63, p 1e-4), but alignment with the EB angle is 0.22-0.30 (p 0.03-0.19 uncorrected;
decoder 0.08-0.22, p >= 0.2). The network's heading code is unrelated to the anatomical layout.

## 2026-10-18 — D35. Anatomical ring and readout, Gate 3, darkness test, real vs rewired

D19's name parsing is gone (supersedes D19 and D27's note that `auto` always means `tuning`). The
`anatomical` ring layout bins the 47 reliable EPGs by EB angle (D32) into 16 wedges; `auto` uses
it when `epg_eb_angles.npz` exists. `heads.decoder: anatomical` replaces the learned 2 x 51 heading
readout with gain * sum r_i (sin a_i, cos a_i) + bias over reliable EPGs, so only gain and bias
are learned. Gate 3 thresholds, fixed before looking: test R^2 > 0.8, mean first-harmonic share
of the 16-wedge profile > 0.5 (about a 53 deg FWHM Gaussian bump), bump-angle vs heading
alignment > 0.8, on data/cache_turns_test and the anatomical layout; tuning layout is reference
only. The darkness test swaps T4/T5 for the flyvis grey steady state for 2 s. That input is not
zero in LPTC space (forward and loom_left about +3.4) but within ~1 sd of the episode-start
(hover) input, so D14's per-subtype baseline doesn't cancel a static scene.

12 BC runs (real/rewire x learned/anatomical readout x seeds 0-2, runs/sweep_turns, 51 min at 3 in
parallel). R^2 is 0.97 everywhere; the anatomical readout costs nothing, for rewired masks too
(ring alignment 0.98 both). No run passes Gate 3: single-bump share 0.15-0.34 per condition (best
run real_anatomical_s2: R^2 0.973, alignment 0.982, share 0.449). The one mask difference outside
seed spread is that share under the anatomical readout (real 0.34 +/- 0.10, rewired 0.19 +/- 0.02).
With the learned readout the real mask has slightly higher R^2 (0.973 vs 0.968, 4.7 vs 5.2 deg).
Topography, ring alignment, darkness drift and closed-loop success don't separate masks at n=3.
A learned readout's heading code is unrelated to EB anatomy (alignment ~0.2, null).

In darkness the trained recurrence holds the decoded heading with a leak: drift ~12 deg at 0.5 s
and 16-42 deg at 2 s, against ~110 deg with recurrent weights zeroed. Anatomical-readout runs fly
worse and very unevenly in closed loop (0.06 to 0.62 per run, SE 0.07) at the same open-loop
action MSE; not explained.

## 2026-10-19 — D36. Real vs rewired at n = 10; bump share definition

Protocol fixed before training (runs/sweep_turns/PROTOCOL.md, 2026-10-18 23:29; copied to
docs/protocols/sweep_turns_n10.md): anatomical readout, seeds 0-9 per mask, primary metric Gate 3
bump share on data/cache_turns_test, difference of means, exact two-sided permutation over all
C(20,10) relabelings, and seeds 3-9 alone as the replication subset since seeds 0-2 produced the
hypothesis. Seeds 3-9 used identical config and data (14 runs, 12 min each, 3 in parallel).

Share = |P1|^2 / sum_{k=1..8} |Pk|^2 on the 16 wedge means; the mean isn't in the denominator, so a
uniform floor can't change it (checked, < 1e-15). The trained cores' floor is ~0.03-0.06 (5th
percentile rate), not softplus(0). A perfect bump with that floor and the measured height reaches
0.5 at FWHM 52.7 deg, same as with no floor, so the threshold is reachable and no corrected metric
is used; a resting-rate-subtracted share is reported alongside. 0.34 is an ideal bump of ~35 deg,
0.19 ~15 deg. The metric isn't monotone at the narrow end: a one-wedge spike and noise both score
~0.13-0.28. scripts/bump_sharpness_check.py, runs/bump_check.

Bump share real 0.299 +/- 0.070, rewired 0.239 +/- 0.046, p 0.030 on all 20, but the seed 0-2
lead (0.337 vs 0.190) doesn't replicate: seeds 3-9 give 0.283 vs 0.260, p 0.37 (resting-rate
version the same). The pooled p leans on the seeds that produced the hypothesis, so it is not a
result. No run passes Gate 3 (best 0.449). R^2 and ring alignment don't separate the masks.
Topography of EPG activity vs EB angle is higher for real (0.50 vs 0.40, p 0.027 all, 0.036 fresh),
uncorrected among six metrics and with the fixed readout pushing both masks toward alignment: a
lead for another round, not a finding. Learned-readout topography per seed: alignment 0.12-0.25,
|rho| <= 0.16, all p >= 0.10, real = rewired. scripts/sweep_stats.py, runs/sweep_turns/stats.json.

## 2026-10-19 — D37. Darkness drift: grey is the wrong no-motion input

The LPTC response to grey (forward and loom_left about +3.4) comes from the outer hexal rings.
D14's baseline is mean relu(grey) over the 263 hexals with radius < 0.6 (rings 0-10), where grey
relu is ~0.02 and cancels. Rings 13-15 (252 hexals) have grey relu up to 1.25, hold 35-58% of
each template's squared weight and give ~94% of the forward and loom_left response. Textured
frames held still give the same offset (within 0.1 of the real-data channel means), so the nets
are trained around it; grey differs from a held frame by up to 0.9 real sd, also inside ring 12.
A per-hexal static reference instead of the scalar baseline would be a constant shift per channel
(pooling is linear after the relu) that InputProjection.bias absorbs exactly, so it can't change
learning or drift. Not adopted.

On grey the decoded heading drifts one way (11 of 12 runs turn right, mean -13.7 deg at 2 s,
mean |drift| 24.9 deg). Holding the last real frame still gives +2.5 and 13.9 deg. The yaw channel
alone explains little (swapping only yaw moves the drift 2.5 of 16 deg; lateral about half): the
nets read rotation from the yaw, lateral and loom pattern together. So about half the grey drift
is calibration, half state-dependent drift toward stable headings near 0 and +-85 deg (the
training dwell headings), the same in all runs and both masks. Darkness tests should hold the last
frame; D35's numbers are the grey case.

EPG time constants (dt/alpha) are 22-67 ms median, 606 ms at most. With W = 0 the ring bump is
gone within 50 ms, so persistence isn't single-neuron. W = 0 removes all synaptic drive (ER -> EPG
too): the ~110 deg jump shows heading lives in coupled activity, not that an attractor holds it;
a feed-forward chain would do the same. Scaling W by 0.9 or 1.1 doubles the 2 s drift and 1.25
runs away, and the velocity field has stable points rather than a continuum: multi-stable at
training headings, not a ring attractor. Real masks have shorter EPG and D7 time constants (34 vs
54 and 34 vs 91 ms, n = 6 each, uncorrected p 0.03 and 0.002). scripts/diagnose_darkness.py,
scripts/diagnose_lptc_grey.py, scripts/plot_time_constants.py, runs/darkness_diag.
