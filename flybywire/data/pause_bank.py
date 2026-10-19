"""Pause bank: T4/T5 input for synthetic pauses in which time stands still (docs/DECISIONS.md D38).

``data/pause_bank_*/``:

* ``bank_fp16.npy``: memmap ``(total_rows, 8, 721)`` float16. Entry ``e`` owns ``hold[e] + rec[e]``
  rows from ``row[e]``: flyvis' response to frame ``t0`` held still for ``hold`` steps, then to
  the real frames ``t0 + 1 ...`` for ``rec`` steps.
* ``entries.npz``: ``episode, t0, hold, rec, row`` per entry, by episode and then by ``t0``.
* ``manifest.json``: the cache the bank belongs to and how close the recovery gets to it.

``t0`` is a decision boundary (``Clocks.decision_core_step``) and ``hold`` a multiple of
``sensory_per_decision``, so a pause inserts whole decisions. The flyvis state at ``t0`` comes from
running the stored hexals from the grey steady state, as the cache builder does, so the entry
transient is real; the recovery runs on from the held state, so the exit transient is real too.
After ``rec`` steps the cached sequence takes over. Entries of one episode are at least
``2 * recovery`` core steps apart, so any subset of them is a consistent set of pauses.

The builder takes a stepper, so this module never imports flyvis.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from flybywire.clocks import Clocks
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory

BANK_FILE = "bank_fp16.npy"
ENTRIES_FILE = "entries.npz"
MANIFEST_FILE = "manifest.json"
ENTRY_KEYS = ("episode", "t0", "hold", "rec", "row")


def draw_entries(lengths, clocks: Clocks, per_episode, hold_steps, recovery, rng) -> dict[str, np.ndarray]:
    """Pause starts uniform over each episode's decision boundaries, at least ``2 * recovery`` steps apart."""
    spd = clocks.sensory_per_decision
    gap = -(-2 * recovery // spd)  # decisions
    lo, hi = hold_steps
    if lo % spd or hi % spd:
        raise ValueError(f"hold_steps {hold_steps} must be multiples of {spd}")
    cols = {k: [] for k in ENTRY_KEYS[:4]}
    for i, n in enumerate(lengths):
        free = n // spd - (per_episode - 1) * (gap - 1)
        if free < per_episode:
            raise ValueError(f"episode {i} ({n} steps) is too short for {per_episode} pauses")
        k = np.sort(rng.choice(free, per_episode, replace=False)) + np.arange(per_episode) * (gap - 1)
        t0 = np.array([clocks.decision_core_step(int(j)) for j in k])
        cols["episode"].append(np.full(per_episode, i))
        cols["t0"].append(t0)
        cols["hold"].append(rng.integers(lo // spd, hi // spd + 1, per_episode) * spd)
        cols["rec"].append(np.minimum(recovery, n - 1 - t0))
    out = {k: np.concatenate(v).astype(np.int64) for k, v in cols.items()}
    out["row"] = np.concatenate([[0], np.cumsum(out["hold"] + out["rec"])[:-1]]).astype(np.int64)
    return out


def build_bank(
    cache: T4T5Cache,
    stepper,
    out_dir: str | Path,
    clocks: Clocks,
    per_episode: int = 4,
    hold_steps: tuple[int, int] = (48, 100),
    recovery: int = 60,
    seed: int = 0,
    batch_size: int = 16,
    branch_size: int = 32,
    manifest_extra: dict | None = None,
    progress: bool = True,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = [cache.length(i) for i in range(len(cache))]
    ent = draw_entries(lengths, clocks, per_episode, hold_steps, recovery, np.random.default_rng(seed))
    n_rows = int((ent["hold"] + ent["rec"]).sum())
    tmp = out_dir / (BANK_FILE + ".tmp")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n_rows, *cache.data.shape[1:]))
    grey = cache.manifest.get("grey_value", 0.5)
    sd = float(np.asarray(cache.data[:: max(1, len(cache.data) // 20000)]).astype(np.float32).std())
    curve, n_curve, repro = np.zeros(recovery), 0, 0.0
    t_start = time.time()
    iterator = range(0, len(cache), batch_size)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, desc="pause bank", unit="batch")
    for start in iterator:
        idx = list(range(start, min(start + batch_size, len(cache))))
        hexals = [Trajectory.load(cache.trajectories[i]).hexals.astype(np.float32) for i in idx]
        x = np.full((len(idx), max(lengths[i] for i in idx), 721), grey, dtype=np.float32)
        for b, h in enumerate(hexals):
            x[b, : len(h)] = h
        x = torch.from_numpy(x).to(stepper.device)
        mine = np.flatnonzero((ent["episode"] >= idx[0]) & (ent["episode"] <= idx[-1]))
        at = {}
        for e in mine:
            at.setdefault(int(ent["t0"][e]), []).append(e)
        snap = {}
        stepper.reset(len(idx))
        for t in range(x.shape[1]):
            out = stepper.step(x[:, t])
            for e in at.get(t, ()):
                b = int(ent["episode"][e]) - idx[0]
                snap[e] = stepper.state[b].clone()
                ref = cache.episode(idx[b])[t].astype(np.float32)
                repro = max(repro, float(np.abs(out[b].half().float().cpu().numpy() - ref).max()))
        for c in range(0, len(mine), branch_size):
            chunk = mine[c : c + branch_size]
            length = int((ent["hold"][chunk] + ent["rec"][chunk]).max())
            seq = np.empty((len(chunk), length, 721), dtype=np.float32)
            for j, e in enumerate(chunk):
                h, t0, m, r = hexals[ent["episode"][e] - idx[0]], ent["t0"][e], ent["hold"][e], ent["rec"][e]
                seq[j, :m] = h[t0]
                seq[j, m : m + r] = h[t0 + 1 : t0 + 1 + r]
                seq[j, m + r :] = h[t0]  # padding, never stored
            stepper.state = torch.stack([snap[e] for e in chunk])
            seq = torch.from_numpy(seq).to(stepper.device)
            out = np.stack([stepper.step(seq[:, s]).to(torch.float16).cpu().numpy() for s in range(length)], 1)
            for j, e in enumerate(chunk):
                m, r, row = ent["hold"][e], ent["rec"][e], ent["row"][e]
                mm[row : row + m + r] = out[j, : m + r]
                if r == recovery:
                    real = cache.episode(ent["episode"][e])[ent["t0"][e] + 1 : ent["t0"][e] + 1 + r]
                    curve += np.abs(out[j, m : m + r].astype(np.float32) - real.astype(np.float32)).mean((1, 2))
                    n_curve += 1
            del out
    mm.flush()
    del mm
    tmp.replace(out_dir / BANK_FILE)
    np.savez(out_dir / ENTRIES_FILE, **ent)
    manifest = {
        "cache": str(cache.dir.resolve()),
        "trajectories": cache.manifest["trajectories"],
        "lengths": lengths,
        "n_entries": len(ent["t0"]),
        "per_episode": per_episode,
        "hold_steps": list(hold_steps),
        "recovery": recovery,
        "seed": seed,
        "total_rows": n_rows,
        "dtype": "float16",
        "cache_sd": sd,
        "recovery_vs_cache_rel": (curve / max(n_curve, 1) / sd).tolist(),
        "rerun_vs_cache_max_abs": repro,
        "build_seconds": round(time.time() - t_start, 1),
        **(manifest_extra or {}),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return out_dir


class PauseBank:
    """Read-only view of a built bank."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.manifest = json.loads((self.dir / MANIFEST_FILE).read_text())
        with np.load(self.dir / ENTRIES_FILE) as z:
            for k in ENTRY_KEYS:
                setattr(self, k, z[k])
        self.data = np.load(self.dir / BANK_FILE, mmap_mode="r")
        self._by_episode = {int(i): np.flatnonzero(self.episode == i) for i in np.unique(self.episode)}

    def __len__(self) -> int:
        return len(self.t0)

    def entries(self, episode: int) -> np.ndarray:
        """Entry ids of one episode, in order of t0."""
        return self._by_episode.get(int(episode), np.empty(0, dtype=np.int64))

    def segments(self, e: int) -> tuple[np.ndarray, np.ndarray]:
        """(hold, rec) T4/T5 rows of entry e, each (n, 8, 721) float16."""
        r, m = int(self.row[e]), int(self.hold[e])
        return self.data[r : r + m], self.data[r + m : r + m + int(self.rec[e])]

    def check_cache(self, cache: T4T5Cache) -> None:
        lengths = [cache.length(i) for i in range(len(cache))]
        if self.manifest["lengths"] != lengths or self.manifest["trajectories"] != cache.manifest["trajectories"]:
            raise ValueError(f"pause bank {self.dir} was not built from cache {cache.dir}")


@dataclass
class PauseSampler:
    """Which pauses go into a training episode: with probability ``p``, ``per_episode`` of its entries."""

    bank: PauseBank
    per_episode: int
    p: float

    def draw(self, episode: int, rng: np.random.Generator) -> np.ndarray:
        cand = self.bank.entries(episode)
        if rng.random() >= self.p or len(cand) == 0:
            return cand[:0]
        return np.sort(rng.choice(cand, min(self.per_episode, len(cand)), replace=False))


def splice(frames, heading, actions, bank: PauseBank, entries, clocks: Clocks):
    """Insert the pauses ``entries`` (ordered by t0) into one episode.

    ``frames`` (T, 8, 721), ``heading`` (T, 2), ``actions`` (K, 4) are the recorded episode. Time
    stands still during a pause: the heading target stays at its t0 value, the frames are the
    bank's, and the pause's decisions get no action target. Decisions and targets after it are
    the recorded ones, shifted by ``hold / sensory_per_decision`` decisions. Returns the spliced
    frames, heading and actions and the action mask (False on inserted decisions).
    """
    spd = clocks.sensory_per_decision
    t0, hold = bank.t0[entries], bank.hold[entries]
    steps = np.insert(np.arange(len(frames)), np.repeat(t0 + 1, hold), np.repeat(t0, hold))
    k0 = np.array([clocks.decision_of_core_step(int(t)) for t in t0])
    dec = np.insert(np.arange(len(actions)), np.repeat(k0 + 1, hold // spd), np.repeat(k0, hold // spd))
    x = np.array(frames[steps])
    amask = np.ones(len(dec), dtype=bool)
    shift = 0
    for e, t, k, m in zip(entries, t0, k0, hold, strict=True):
        h, r = bank.segments(e)
        a = t + 1 + shift
        x[a : a + m] = h
        x[a + m : a + m + len(r)] = r
        d = k + 1 + shift // spd
        amask[d : d + m // spd] = False
        shift += m
    return x, heading[steps], actions[dec], amask
