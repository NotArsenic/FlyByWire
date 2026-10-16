# data/

Everything under `data/` except this file is git-ignored. `FBW_DATA_DIR` (see `.env`)
points here by default. On WSL, keep this on the Linux filesystem, never under `/mnt/c`.

| Path | Contents | How it gets populated |
|---|---|---|
| `flyvis/` | `FLYVIS_ROOT_DIR`: flyvis pretrained models, `direction_map.json` | `flyvis download-pretrained --skip_large_files` (run by `make env`); `make calibrate` writes `direction_map.json` |
| `flywire/raw/` | FlyWire Codex connections + cell-type annotation tables (snapshot 783) | **Manual download** from FlyWire Codex (sign-in / terms acceptance may be required); see the README "Data" section |
| `flywire/processed/` | `cx_graph.npz`: mask `M` (bool, `M[i, j]` = pre i → post j), neuron ids, type labels, group index arrays, instance names | `python -m flybywire.connectome.flywire` |
| `expert/` | `traj_XXXXX.npz`: `hexals (T, 721)` fp16 @ 100 Hz, `actions (T_dec, 4)` fp32 normalized, state arrays, `rel_yaw (T,)`, `yaw_rate (T,)`, `collided`, metadata | `scripts/collect_expert.py` (week 1) |
| `frames/` | raw frames for at most 20 episodes (debugging and video only) | `scripts/collect_expert.py --save-frames` |
| `cache/` | `t4t5_fp16.npy` memmap `(total_T, 8, 721)`, `offsets.npy`, `manifest.json` | `scripts/build_cache.py` |
| `smoke/` | synthetic trajectories + cache for `make smoke` (plumbing only, not a result) | `scripts/smoke_end_to_end.py` |
