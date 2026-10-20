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

## 2026-10-19 — D38. DART noise instead of textures; two DAgger rounds; real vs rewired again

Closed-loop failures in the D35 runs (600 episodes per condition) were mostly late obstacle
avoidance (onset ~1.5 m, expert 2.1 m; 195/600 for real/learned) and, in the bad anatomical runs,
a +27 deg yaw bias into the +y wall (304/600 for real/anatomical), not heading drift. A
160-episode texture pilot (ridge and MLP probes, runs/texture_pilot) showed no gain from textured
walls or obstacles, and wall textures lowered lateral-position and heading decoding. Not adopted;
env.obstacle_texture stays (default none) so the pilot can be rerun.

Instead expert.exec_noise (DART, Laskey et al. 2017): OU noise on the executed action (sd
0.1/0.25/0/0.8 normalised, tau 0.5 s, per-episode scale U(0,1)), label is the clean expert action.
data/expert_dart (seeds 0-999) and expert_dart_test (2000-2099), untextured. Real/learned seeds
0-2: 52.8% -> 76.2% closed loop. Gate 2 on it is 0.615 (was 0.880), not from textures: T4/T5 lag
yaw rate by ~80 ms and DART yaw rate changes 3.4x faster per frame, so yaw rate 80 ms ago explains
0.71 of the current one (0.94 before); ridge against the 80 ms-shifted target gives 0.83 (0.92).
Decoded heading test R^2 is ~0.92 instead of 0.975.

DAgger from real/learned seed 0 (plain labels): 59.5% -> 75.0% -> 76.0% (n = 200); wall hits
vanish, obstacle hits stay ~48. Stopped after round 2. Seed 0 was the weakest reference seed
(seeds 1, 2: 87.5, 81.5% without DAgger) and real/learned s0-2 on the aggregate average 65.8%, so a
gain over DART alone isn't shown.

Sweep on data/cache_dart_agg2 (1600 episodes), protocol docs/protocols/sweep_dart_agg2.md:
anatomical seeds 0-5 per mask (6-9 cut by the protocol's time rule after a crash cascade forced a
retrain wave). Bump share real 0.322 vs rewired 0.322, p 0.998; the D36 lead is gone. No run passes
Gate 3 (best 0.363). R^2 0.92 and ring alignment 0.95 everywhere. Closed loop: the anatomical
readout now flies like the learned one (74.5% vs 76.4%) and its wall failure is gone. The rewired
mask flies better than the real one (83.6% vs 67.3%, post hoc p 0.015). Demo candidate
real_anatomical_s5: share 0.363, R^2 0.932, 79.5% closed loop (picked from six, so optimistic).
runs/sweep_dart, runs/gate3_dart, runs/dagger_dart, runs/failure_breakdown.

## 2026-10-19 — D39. Time-freeze pauses in BC

D37 showed multi-stable heading under a held frame. Blanking input inside recorded windows while
keeping the heading targets would teach hallucinated turns (there is no efference copy), so the
pauses are synthetic and time stands still. bc.pauses {bank, per_episode, p} splices them into
training episodes only (key absent = old behaviour; 24 updates on cache_turns give identical
losses and grad norms). A pause starts after core step 4k+3 (k uniform, so mid-maneuver too),
lasts M = 48..100 steps (multiple of 4) and shifts every later decision by M/4. Heading target
held at theta(t0), action loss off on the M/4 inserted decisions, aux heading loss on, state
carried through TBPTT windows. Recovery decisions keep their targets.

Input is real flyvis, precomputed (scripts/build_pause_bank.py, flybywire/data/pause_bank.py):
rerun the stored hexals, branch the state at t0, hold the frame M steps, then continue on the
real frames t0+1.. for R = 60 steps before the cache takes over. R = 60, not 30: mean |bank -
cache| / cache sd is 2.1% after 30 steps and 0.24% after 60. 4 entries per episode, >= 2R apart.
data/pause_bank_turns: 4000 entries, 6.0 GB, 294 s.

Pilot on cache_turns (anatomical readout, seed 0, 2 pauses per episode, n = 1 per cell): test R^2
and error unchanged (0.975 -> 0.976 real, 0.971 -> 0.978 rewired). Drift through a held frame with
its real transient, mean |.|: 1 s 8.9 -> 5.9 real, 9.5 -> 5.9 rewired; 2 s 11.6 -> 8.8, 13.2 ->
8.9; 10 s 13.6 -> 14.2, 25.2 -> 12.9. Persistence improves inside the trained length, not beyond
it for the real mask. The velocity field is ~2x flatter by rms but keeps its stable points near
-79, +5, +87 deg: attenuated, not a continuous attractor. Grey drift gets worse (2 s 22.8 -> 37.2
real, 18.2 -> 51.1 rewired), unexplained. Real and rewired don't separate (drift, field rms; bump
share 0.37 vs 0.30, within seed spread; Gate 3 still fails). Persistence is now trained, so only
real vs rewired under the same training is a comparison. The pilot ran on the pre-DART turning
data; the next run should use data/cache_dart_agg2 with longer holds. runs/step4_pilot.

## 2026-10-19 — D40. Cache cleanup; closed-loop intervals over seeds

Deleted data/cache (300 plain episodes), cache_turns, cache_dart, cache_dagger_r1_off and
pause_bank_turns (44 GB). The hexals they were built from are kept, so each is one build_cache.py /
build_pause_bank.py away; cache_dart_agg2 (the demo candidate's training data) and the test caches
stay. test_closed_loop now reads cache_dart_test, since data/cache is gone.

The README's closed-loop intervals were Wilson intervals over 600 pooled episodes, which treats
episodes from the same network as independent and ignores the spread between seeds (real/learned
runs at 76.0, 45.0, 76.5%). The table now gives the per-seed rates and a t interval over the three
seeds. The D38 "rewired flies better" p of 0.015 was already a run-level test (exact permutation,
12 runs, 924 relabelings); permuting mask labels only within readout gives the same 0.015 (6 of
400). It stays post hoc; docs/protocols/sweep_uniform_pauses.md tests it on fresh seeds (H3).

## 2026-10-19 — D41. Lesions in closed loop on the demo candidate

scripts/eval_lesions.py, real_anatomical_s5 (model_best.pt), the 200 held-out seeds 10000-10199,
lesion on for the whole flight. Ring on the anatomical layout after a 50-step washout; "gain" is the
bump's turn against the drone's turn over 0.5 s windows with more than 20 deg of turning.

| lesion | success [Wilson 95] | heading err (deg) | EPG rate | bump share | gain |
|---|---|---|---|---|---|
| none | 79.5% [73.4, 84.5] | 5.8 | 1.18 | 0.46 | 1.12 |
| EPG | 0.0% [0.0, 1.9] | 59.7 | 0 | 0 | 0 |
| PEN | 82.5% [76.6, 87.1] | 11.0 | 1.25 | 0.47 | 1.23 |
| D7 | 5.0% [2.7, 9.0] | 6.7 | 1.33 | 0.42 | 0.81 |

EPG: the bump is gone, and so is steering. The action becomes nearly constant (sd 0.03-0.05 per
dim against 0.14-0.29 intact) with a steady left turn and drift, and every flight ends on a side
wall after ~1.5 s. PFL3, the only thing the actor reads, gets its input through EPG in this
network, so the policy is not a reactive map from vision to action that bypasses the compass.

PEN: the prediction that a PEN lesion freezes the compass is wrong for this model. The bump stays,
keeps its share and follows the turns (gain 1.23); heading error doubles and flight is unchanged.
Self-motion reaches EPG continuously through the ER input, so the heading is driven by vision
rather than shifted by PEN. That fits D37/D39: no internal integrator holds the heading.

D7: the compass is fine (error 6.7 deg, gain 0.81, bump a little broader) but the drone flies
faster (forward command mean +0.68 normalised against +0.09) and hits walls and obstacles.
D7 matters for the motor side here, not for the ring. runs/lesions/real_anatomical_s5.

## 2026-10-20 — D42. Uniform holding headings; last real-vs-rewired sweep

expert.maneuvers.targets=uniform: the far heading of a maneuver is drawn from U[-180, 180) and held
(hold_s [1, 3], wait_s [0.5, 2], DART noise as D38). 6/1000 expert collisions, as for the old
maneuvers. Protocol docs/protocols/sweep_uniform_pauses.md, written and committed before collection.
Anatomical readout, pause training with 1-3 s holds; uniform arm real/rewired seeds 10-15 on
data/cache_uniform, control arm seeds 10-12 on data/cache_dart (old maneuvers), all scored on the
uniform test set. Mean (sd) per cell:

| cell | drift 2 s (deg) | field rms (deg/s) | bump share | closed loop |
|---|---|---|---|---|
| uniform real (6) | 9.54 (0.51) | 3.47 (0.71) | 0.285 (0.064) | 64.3% (25.0) |
| uniform rewired (6) | 9.78 (0.84) | 4.14 (1.19) | 0.286 (0.094) | 82.9% (8.2) |
| control real (3) | 9.74 (0.53) | 4.37 (1.17) | 0.364 (0.057) | |
| control rewired (3) | 9.15 (0.17) | 5.06 (0.61) | 0.214 (0.049) | |

H1 not supported: field rms 0.91 deg/s lower with uniform holds, p 0.074, and a stable point near
+-85 deg is still there in 8 of 12 uniform runs (5 of 6 control); every run keeps one near 0. Holding
headings everywhere flattens the field a little but does not turn the stable points into a ring.
H2 not supported: drift p 0.55, field rms p 0.26, bump share p 0.996 (real - rewired -0.001); Holm
p >= 0.78. H3 does not replicate at p < 0.05: rewired flies better again (+18.6 points, p 0.119),
same direction as D38 with a spread driven by two weak real seeds (26%, 46%).

That ends the wiring comparison: across the turning, DART and uniform-hold data, under the same
training, the FlyWire mask has not beaten a degree-preserving rewire on any pre-registered measure.
One scoring step (uniform_rewire_s10, pause eval) had failed on CUDA OOM six times while post.sh
marked the run done anyway; it was rerun alone with the same command before the tests.
runs/sweep_uniform, stats.json.

## 2026-10-20 — D43. PPO fine-tuning does not beat the BC policy

flybywire/train/ppo.py from real_anatomical_s5 (BC, 79.5% on test seeds 10000-10199): KL anchor to
the frozen BC policy, aux heading loss on, critic warm-up with the actor frozen, time-limit ends
bootstrapped from final_obs. Three setups, all scored on deterministic flights (action mean):

- collision -10, exploration std 0.22, rollouts of 128 decisions: test 76.0% at update 20, 70.0%
  at 80. The policy flew faster (mean progress 9.27 -> 9.61 m) and crashed more: the progress
  reward outweighed a -10 crash.
- collision -100 and checkpoints picked on 100 validation seeds (20000..), never the test seeds:
  validation 75% (BC), 74, 64, 63% at updates 39, 79, 119.
- exploration std 0.1, rollouts of 256 decisions, KL floor 0.05: validation 75% (BC), 76, 74, 73,
  71% at updates 19-99. The best checkpoint (update 19) scores 75.0% on the test seeds.

Success of the noisy training flights held at ~75-82% throughout while the action mean got worse:
PPO improves the stochastic policy, and the deterministic one drifts away from BC (KL to BC up to
0.3-0.4) without gaining. Stopped well inside the 3-day budget; the BC policy stays the model of
record. reward.collision is -100 and ppo.log_std -2.3 from here on.

Side issues: a training env opened in a pybullet GUI died when its window was closed and took the
run with it, so the live view's chase camera is now a copy of env 0 in the viewer's own process.
PPO with its validation envs, three BC trainings and closed-loop scoring together ran the 31 GB
machine out of memory; they now run one after another. runs/ppo.

## 2026-10-20 — D44. Dense-corridor stage

A last training stage on harder corridors. The default layout spreads 8 obstacles over 26 m with
+-0.5 m jitter; packed tighter that way, neighbours nearly touch and the expert, which plans around
one obstacle at a time, crashes (12 obstacles 5/40, 16: 17/40). env.obstacle_spacing_m places
obstacles at a fixed spacing from x = 4 m with +-15% jitter (key absent or null = old layout, same
random draws). Levels in configs/default.yaml curriculum: spaced 2.5 m, dense 2.0 m, slalom 1.6 m;
the last two with a slower, steeper expert (speed 1.0, max heading 65 deg, influence 0.5 m), which
flies all three at 96.5-99% on held-out seeds. Gap height (a Flappy Bird stage) was not tried: the
8 LPTC channels have no up/down looming, so the network couldn't see where a gap is.

400 DART expert episodes per level (seeds 3000-3399) plus the 1000 data/expert_dart episodes as
replay, data/cache_curriculum (2200 episodes, 25 GB). BC fine-tuned from real_anatomical_s5 (bc
--init), lr 3e-4, 20 epochs; model.pt (last epoch) was fixed as the stage's output before any flight.
Closed loop, test seeds 10000-10199, success [Wilson 95]:

| | default | spaced | dense | slalom |
|---|---|---|---|---|
| expert | 100 | 96.5 | 99.0 | 98.0 |
| real_anatomical_s5 (BC, stage 3) | 79.5 [73.4, 84.5] | 65.5 [58.7, 71.7] | 58.0 [51.1, 64.6] | 54.5 [47.6, 61.3] |
| dense stage (stage 4) | 80.5 [74.5, 85.4] | 67.5 [60.7, 73.6] | 65.0 [58.2, 71.3] | 70.0 [63.3, 75.9] |

Slalom gains 15.5 points with no loss on the default corridor; spaced and dense move within their
intervals. One network, one seed. Earlier stages on the same seeds (separate networks, not one
lineage except stage 4 from 3): untrained 0 / 0, BC on the plain expert 53.5 / 25.5, BC on turns
59.0 / 31.5 (default / slalom). docs/media/progress.gif shows each flying held-out slalom seed
10001, the first seed where stage 4 makes it and stage 3 does not (so picked to show a difference;
the rates in the labels are the measured ones). runs/curriculum.

Lesions on the stage-4 network (scripts/eval_lesions.py, default corridor, 200 test seeds): none
80.5%, EPG 0%, PEN 65.0% (heading error 21.9 deg against 4.4, gain 0.71 against 1.01), D7 3.5%.
Unlike stage 3 (D41), PEN now matters: the bump follows turns less well without it and success
drops 15.5 points. The compass still isn't frozen. docs/media/dashboard.gif shows the stage-4
network on slalom seed 10001 with EPG silenced from 6 s; a run with PEN silenced at 4-7 s ended on
an obstacle at 7.7 s, before its planned EPG window, and was not used. runs/lesions/dense_s5.

## 2026-10-20 — D45. Caches deleted; recordings streamed and capped

Deleted every T4/T5 cache and pause bank except data/cache_dart_test (the closed-loop test reads
it): cache_curriculum, cache_dart, cache_dart_agg2, cache_uniform(_test), cache_turns_test and all
pause_bank_* (about 97 GB). All are rebuilt from the stored hexals with scripts/build_cache.py and
scripts/build_pause_bank.py; the trajectories, runs and checkpoints are kept.

The PPO window used to hold a whole recorded episode in memory before encoding (1600x840 RGB, about
4 MB a frame). It now streams frames to the encoder. ppo.py checks room_to_record before each
recording and skips it once <out>/watch holds --record-budget-gb (default 2 GB) or the disk has
under 10 GB free. A 10 s episode is 250 frames, a few MB of H.264.

PPO from the stage-4 network (runs/curriculum/dense_s5/model.pt) with --watch --record-every 50,
default 400 updates. The first launch flew the default corridor: the dense stage applied its
corridor overrides only when the training data was collected, so the checkpoint's config still
describes the default layout, and load_bc takes the corridor from it. Stopped at update 15
(runs/ppo/dense_s5_default, kept with its update-0 recording). Relaunched with
env.obstacle_spacing_m=2.0 env.n_obstacles=12, so training and the validation that picks
model_best.pt are both on the dense level: runs/ppo/dense_s5_dense.
Stopped the dense run at update 29 (validation 0.67 at the start, 0.66 at update 19): the corridor
still looked sparse, see D46.

## 2026-10-20 — D46. Forest level: rows of one or two pillars

The dense levels looked sparse in flight because a 10 s episode at about 1 m/s ends near
x = 10-12 m, so only the first 4 or so of the 12-14 obstacles are ever reached. env.pair_prob > 0
switches the layout to rows at obstacle_spacing_m, each holding one pillar (as before, |y| <= 0.7)
or, with that probability, two pillars side by side at y in [-1.3, 1.3], resampled until they
don't overlap and one gap (between them or to a wall) is at least MIN_GAP = 1.1 m. n_obstacles
counts pillars. curriculum.forest: spacing 1.6, 20 pillars, pair_prob 0.5, the dense expert
settings. A flight now passes 8-10 pillars.

The expert used to pick a side of the single nearest obstacle, which with two pillars at one x
could steer into the other. goal() now takes every obstacle within ROW_TOL = 0.3 m (along x) of
the nearest one ahead, including ones already passed, and aims at the widest gap of that row.
With one obstacle per row this is the old rule exactly, tie-break included: layouts and goal()
for 200 seeds x 50 positions on the default, dense and slalom levels are bit-identical before and
after. Neighbours on those levels are at least 1.1 m apart along x, so no row ever has two.

200 held-out forest corridors (seeds 10000..): expert 98.0%, stage-4 network 65.0%
(runs/curriculum/eval_forest.json). PPO from stage 4 on the forest level, --watch --record-every
50: runs/ppo/dense_s5_forest.
The forest PPO run finished its 400 updates: validation 0.68 (BC) -> 0.60-0.73 with no trend,
best 0.73 at update 319 on 100 seeds, inside the noise. BC stays the model of record.

## 2026-10-20 — D47. The dodge side is in T4/T5 but partly lost in the 8 LPTC channels

In the dashboard (20 s flights, now the checkpoint's own 10 s and a level picker) the policy flew
straight into pillars; every crash had the pillar within 0.0-0.3 m of the drone's y. Question: is
the side to dodge to visible to the network? scripts/probe_pillar_dodge.py on the T4/T5 activity
of expert episodes (400 default from data/expert_dart, 400 new forest DART episodes in
data/expert_forest, seeds 3000-3399), decisions with the nearest row 1.0-2.5 m ahead and heading
within 30 deg: 53,183 decisions, 26% with a pillar within 0.35 m of the drone's y ("dead ahead").
Target: sign of the expert's world-frame lateral velocity command (|vy| > 0.1 m/s; majority 50%).
Held-out episodes (every 4th), accuracy all / dead ahead:

| features | linear | MLP |
|---|---|---|
| geometry (privileged pillar offsets) | 79.9 / 85.0 | 88.1 / 91.2 |
| lptc8 (the stage-4 network's channels) | 70.7 / 70.5 | 78.0 / 77.3 |
| lptc8, lags 0-150 ms | 72.3 / 71.8 | 80.3 / 81.1 |
| 6 self-motion + 6 loom sectors (edges +-18, +-36 deg) | 73.6 / 72.2 | 79.7 / 80.1 |
| 6 self-motion + 8 loom sectors | 73.8 / 71.8 | 79.9 / 80.4 |
| 8 loom sectors, lags 0-150 ms | 74.2 / 72.4 | 81.9 / 83.0 |
| T4/T5, 8 x 721 | 84.2 / 87.4 | 85.7 / 87.3 |

Forest dead ahead alone: lptc8 67.4 / 75.2, T4/T5 84.3 / 82.7. A linear readout of T4/T5 gets
the side about as well as the pillar positions do; the fixed 8-channel pooling loses 10-17
points of it. Azimuth-binned loom templates win back only 2-3: the information is not in
flow-field-shaped sums. A single learned linear template on T4/T5 carries it, which is what
lptc.mode=full (learnable templates, initialised from the current ones) allows; never tried.
R^2 of the v_lateral / yaw-rate commands is low for every set (geometry MLP 0.52 / 0.32: the
expert's OU wander and DART noise are not predictable), T4/T5 linear 0.47 / 0.38, lptc8 linear
0.13 / 0.16. The T4/T5 cache holds flyvis output before the pooling, so changing the templates
needs no new cache; one is needed only because the curriculum caches were deleted (D45): 0.17 s
per episode from stored hexals (800 in 137 s). runs/probe.
