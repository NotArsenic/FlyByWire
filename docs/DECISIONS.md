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
