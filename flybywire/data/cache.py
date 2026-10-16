"""T4/T5 activity cache: ``data/cache/{t4t5_fp16.npy, offsets.npy, manifest.json}``.

* ``t4t5_fp16.npy`` — memmap ``(total_T, 8, 721)`` float16, trajectories concatenated in manifest
  order; episode ``i`` occupies rows ``offsets[i]:offsets[i + 1]``.
* ``offsets.npy`` — ``(n_traj + 1,)`` int64.
* ``manifest.json`` — flyvis model id, dt, cell types, config hash, trajectory file list.

The builder runs a stepper (normally :class:`~flybywire.vision.flyvis_stepper.FlyvisStepper`) over
batches of 10–20 trajectories, initialised from the grey steady state, slicing out T4/T5 at every
step (full 45,669-cell responses are never held). Shorter trajectories in a batch are padded
with grey after their end; padded outputs are discarded. One cache serves every mask condition
and seed. BC reads only this cache and never runs flyvis.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from flybywire.data.trajectories import Trajectory

CACHE_FILE = "t4t5_fp16.npy"
OFFSETS_FILE = "offsets.npy"
MANIFEST_FILE = "manifest.json"


class Stepper(Protocol):
    device: torch.device

    def reset(self, batch_size: int | None = None, env_ids=None) -> None: ...
    def step(self, hexals: torch.Tensor) -> torch.Tensor: ...


def build_cache(
    traj_paths: Sequence[str | Path],
    out_dir: str | Path,
    stepper: Stepper,
    manifest_extra: dict | None = None,
    batch_size: int = 16,
    grey_value: float = 0.5,
    n_types: int = 8,
    n_hexals: int = 721,
    progress: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_paths = [Path(p) for p in traj_paths]
    lengths = []
    for p in traj_paths:
        with np.load(p, allow_pickle=False) as z:
            lengths.append(int(z["rel_yaw"].shape[0]))
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    total = int(offsets[-1])
    tmp = out_dir / (CACHE_FILE + ".tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(total, n_types, n_hexals))
    t0 = time.time()
    iterator = range(0, len(traj_paths), batch_size)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="cache", unit="batch")
    for start in iterator:
        idx = list(range(start, min(start + batch_size, len(traj_paths))))
        hex_list = [Trajectory.load(traj_paths[i]).hexals for i in idx]
        tmax = max(h.shape[0] for h in hex_list)
        batch = np.full((len(idx), tmax, n_hexals), grey_value, dtype=np.float32)
        for b, h in enumerate(hex_list):
            batch[b, : h.shape[0]] = h.astype(np.float32)
        x = torch.from_numpy(batch).to(stepper.device)
        stepper.reset(len(idx))
        out = np.empty((len(idx), tmax, n_types, n_hexals), dtype=np.float16)
        for t in range(tmax):
            out[:, t] = stepper.step(x[:, t]).to(torch.float16).cpu().numpy()
        for b, i in enumerate(idx):
            mm[offsets[i] : offsets[i + 1]] = out[b, : lengths[i]]
        del out
    mm.flush()
    del mm
    tmp.replace(out_dir / CACHE_FILE)
    np.save(out_dir / OFFSETS_FILE, offsets)
    manifest = {
        "trajectories": [str(p.resolve()) for p in traj_paths],
        "lengths": lengths,
        "total_T": total,
        "shape": [total, n_types, n_hexals],
        "dtype": "float16",
        "grey_value": grey_value,
        "build_seconds": round(time.time() - t0, 2),
        **(manifest_extra or {}),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return out_dir


class T4T5Cache:
    """Read-only view of a built cache."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.manifest = json.loads((self.dir / MANIFEST_FILE).read_text())
        self.offsets = np.load(self.dir / OFFSETS_FILE)
        self.data = np.load(self.dir / CACHE_FILE, mmap_mode="r")
        if self.data.shape[0] != self.offsets[-1]:
            raise ValueError("cache rows do not match offsets")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def length(self, i: int) -> int:
        return int(self.offsets[i + 1] - self.offsets[i])

    def episode(self, i: int) -> np.ndarray:
        """(T_i, 8, 721) float16 memmap view."""
        return self.data[self.offsets[i] : self.offsets[i + 1]]

    @property
    def trajectories(self) -> list[Path]:
        return [Path(p) for p in self.manifest["trajectories"]]

    def check_compatible(self, model_id: str | None = None, dt: float | None = None) -> None:
        if model_id is not None and self.manifest.get("flyvis_model") not in (None, model_id):
            raise ValueError(f"cache built with {self.manifest.get('flyvis_model')}, config wants {model_id}")
        if dt is not None and self.manifest.get("dt") is not None and abs(self.manifest["dt"] - dt) > 1e-12:
            raise ValueError(f"cache built with dt={self.manifest['dt']}, config wants {dt}")
