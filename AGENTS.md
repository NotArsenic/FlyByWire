# AGENTS.md — conventions for agent sessions on Fly-By-Wire

Read this before changing code. The full spec lives in the README and `docs/DECISIONS.md`.
**Log every deviation, surprise or non-obvious choice in `docs/DECISIONS.md`** (dated entry + why).

## Hard rules

- **fp32 only.** No AMP / autocast / bf16 / `torch.compile`. On-disk caches may be fp16 but are upcast
  on load. (Ring-attractor dynamics are marginally stable; low precision makes fake drift.)
- **BC never runs flyvis.** BC reads only the T4/T5 cache in `data/cache/` (a test enforces that
  `flybywire.train.bc` does not import flyvis).
- **Import flyvis only via `flybywire.paths.import_flyvis()`** and run flyvis calls inside
  `with torch.device(dev):`. flyvis resolves `FLYVIS_ROOT_DIR` against the CWD and sets torch's
  global default device to CUDA on import (DECISIONS D11/D12).
- The frozen flyvis stepper (`vision/flyvis_stepper.py`) is the gradient boundary; its default
  `fast` mode uses private `Network._next_state` — rerun `make gate1` after any flyvis upgrade.
- Do not guess flyvis / gym-pybullet-drones / gymnasium APIs: read the installed source.
- Environment: Python 3.12, torch cu130 (Blackwell `sm_120`), conda env `flybywire`.
- Never use the `toy` connectome for reported results (it logs a loud warning).
- No novelty claims ("first", "novel", "state-of-the-art") anywhere.
- No `sudo`; do not edit files outside the repo. Ask before jobs longer than ~10 min.

## Connectome convention

`M[i, j] = 1` iff **presynaptic i → postsynaptic j**. Dynamics use `r @ (W * M)` (row = pre,
column = post). `W` is a dense `(N, N)` parameter times a registered boolean buffer `M`; no sparse
tensors. Group index sets (ER, EPG, PEN, PEG, D7, PFL3, ...) come from `configs/default.yaml`.

## Clocks and decision alignment

| Layer | Rate | Steps per decision |
|---|---|---|
| PyBullet physics | 200 Hz | 8 |
| Render + flyvis (dt = 1/100) + CX core | 100 Hz | 4 |
| Actor readout | 25 Hz | 1 |

Defined once in `flybywire/clocks.py` (`Clocks`). Core steps are indexed `t = 0..T-1`; decision `k`
is read out **after** core step `t = 4k + 3` (`Clocks.decision_core_steps`), i.e. after the last
sensory tick of its window, and the action is held for the next 4 core steps / 8 physics steps.
Data collection, BC and PPO must all use `Clocks` — never re-derive the alignment. Episodes have
`T = 4K` frames; `reset()` runs a 4-frame hover pre-roll and the env returns 4 frames per step.
PPO uses gymnasium `AutoresetMode.SAME_STEP` (DECISIONS D20).

## Invariants (each has a test in `tests/`)

1. The only neuron↔neuron path in the core is `r @ (W * M)` (no LayerNorm/attention/dense over N).
2. `dL/dW == 0` wherever `M == 0`.
3. Input projection output is exactly 0 on every non-ER neuron.
4. Actor reads only PFL3 rates; heading decoder reads only EPG rates; critic reads `r.detach()`.
5. flyvis gets no gradients (`requires_grad == False` everywhere; `no_grad` + eval).
6. Lesioned types have rates exactly 0 at every step; EPG lesion ⇒ heading decoder output == bias.
7. All mask conditions share N and the exact cell-type index sets; only edges differ.
8. Rewire preserves every in-/out-degree exactly, no self-loops / duplicates, same edge count.
9. Heading target = `(sin, cos)(wrap(yaw_t − yaw_0))`.
10. All parameters and buffers are fp32.

## Data locations

- `data/flyvis/` — `FLYVIS_ROOT_DIR` (pretrained models, `direction_map.json`)
- `data/flywire/raw/` (manual Codex download, snapshot 783), `data/flywire/processed/cx_graph.npz`
- `data/expert/traj_XXXXX.npz` — hexals fp16 @ 100 Hz, actions @ 25 Hz, privileged state
- `data/cache/` — `t4t5_fp16.npy` memmap `(total_T, 8, 721)`, `offsets.npy`, `manifest.json`
- `runs/` — checkpoints, logs, reports. Never put data under `/mnt/c` on WSL.

## Commands

`make env` (create env) · `make check` · `make test` · `make smoke` · `make gate1` ·
`make calibrate` · `make params` · `make lint` · `make lock`

After modifying code, run `graphify update .` to refresh the knowledge graph (git-ignored).
