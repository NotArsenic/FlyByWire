#!/usr/bin/env python
"""Why closed-loop flights fail: what ended each episode, where, and what the drone was doing.

fly         fly held-out seeds with each checkpoint and save the states (<run>/failure_flights.npz)
fly-expert  the same for the scripted expert, using the config of a checkpoint
report      tables per run and per condition from the saved flights

The end of an episode is the surface nearest to the last position: an obstacle, a side wall, the
floor or the ceiling (the env only says "collision"). Heading drift: the decoded heading is more
than --drift-deg off the true one for --drift-s seconds in a row within the last --tail-s seconds.
An approach is the drone coming within 3 m of an obstacle's front face along x; it is on course
when its lateral distance to the obstacle centre is under half width + 0.3 m at that point.

Usage:
  python scripts/failure_breakdown.py fly RUN [RUN ...] [--n 200] [--seed 10000] [--ckpt model_best.pt]
  python scripts/failure_breakdown.py fly-expert --like RUN --out DIR [--n 200] [--seed 10000]
  python scripts/failure_breakdown.py report RUN [RUN ...] [--expert DIR] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from flybywire.envs.expert import surface_distance
from flybywire.eval.closed_loop import ExpertDriver, PolicyDriver, fly, load_checkpoint
from flybywire.eval.metrics import wilson_interval

FLIGHTS = "failure_flights.npz"
SAMPLE = slice(3, None, 4)  # the last frame of each decision window
DISTANCES = (3.0, 2.0, 1.5, 1.0, 0.5)
ENDS = ("obstacle", "side_wall", "floor", "ceiling", "out_of_bounds", "time", "end")
LATERAL_ONSET_M = 0.15


def pack(episodes) -> dict:
    """Decision-rate states of the episodes, zero padded to the longest."""
    n, k = len(episodes), max(e.n_decisions for e in episodes)
    out = {
        "seeds": np.array([e.seed for e in episodes]),
        "reasons": np.array([e.reason for e in episodes]),
        "length": np.array([e.n_decisions for e in episodes]),
        "obstacles": np.stack([e.obstacles for e in episodes]),
    }
    fields = {"pos": 3, "vel": 3, "rel_yaw": 0, "yaw_rate": 0}
    if episodes[0].heading_err_deg is not None:
        fields["heading_err"] = 0
    for name, width in fields.items():
        out[name] = np.zeros((n, k, width) if width else (n, k), np.float32)
        for i, e in enumerate(episodes):
            src = e.heading_err_deg if name == "heading_err" else e.state[name]
            out[name][i, : e.n_decisions] = src[SAMPLE]
    return out


def end_kind(
    reason: str, pos, obstacles, half_width: float, height: float
) -> tuple[str, int | None]:
    """What ended the episode, and for an obstacle which row of ``obstacles``."""
    if reason != "collision":
        return reason, None
    near = [surface_distance(pos[0], pos[1], ob)[0] for ob in obstacles]
    dist = {
        "side_wall": half_width - abs(pos[1]),
        "floor": pos[2],
        "ceiling": height - pos[2],
        "obstacle": min(near),
    }
    kind = min(dist, key=dist.get)
    return kind, int(np.argmin(near)) if kind == "obstacle" else None


def sustained(flag: np.ndarray, n: int) -> bool:
    """True if ``flag`` has a run of at least n Trues."""
    run = 0
    for f in flag:
        run = run + 1 if f else 0
        if run >= n:
            return True
    return False


def approaches(pos, vel, obstacles, length: int, hit: int | None) -> list[dict]:
    """One row per obstacle the drone came within 3 m of (front face, along x)."""
    rows = []
    rank = np.argsort(np.argsort(obstacles[:, 1])) + 1
    for j, (_, ox, oy, hx, hy) in enumerate(obstacles):
        d = (ox - hx) - pos[:length, 0]
        near = np.flatnonzero(d <= DISTANCES[0])
        if not len(near):
            continue
        k0 = near[0]
        row = {
            "obstacle": j,
            "rank": int(rank[j]),
            "hit": hit == j,
            "on_course": bool(abs(pos[k0, 1] - oy) < hy + 0.3),
        }
        for dist in DISTANCES:
            k = np.flatnonzero(d <= dist)
            # the first obstacle is already within 3 m at the start, before there is a sample at that distance
            k = k[0] if len(k) and (k[0] > 0 or d[0] >= dist) else None
            row[f"clearance_{dist}"] = (
                None if k is None else float(abs(pos[k, 1] - oy) - hy)
            )
            row[f"vx_{dist}"] = None if k is None else float(vel[k, 0])
        end = int(np.flatnonzero(d > 0)[-1]) + 1 if (d > 0).any() else k0
        moved = np.flatnonzero(np.abs(pos[k0:end, 1] - pos[k0, 1]) > LATERAL_ONSET_M)
        row["onset_m"] = float(d[k0 + moved[0]]) if len(moved) else None
        rows.append(row)
    return rows


def episode_rows(
    fl: dict,
    half_width: float,
    height: float,
    drift_deg: float,
    drift_s: float,
    tail_s: float,
    control_hz: int = 25,
) -> list[dict]:
    rows = []
    for i, seed in enumerate(fl["seeds"]):
        k = int(fl["length"][i])
        pos, vel, obstacles = fl["pos"][i, :k], fl["vel"][i, :k], fl["obstacles"][i]
        kind, hit = end_kind(
            str(fl["reasons"][i]), pos[-1], obstacles, half_width, height
        )
        row = {
            "seed": int(seed),
            "end": kind,
            "decisions": k,
            "x_end": float(pos[-1, 0]),
            "failed": kind not in ("time", "end"),
            "rel_yaw_end_deg": float(np.degrees(fl["rel_yaw"][i, k - 1])),
        }
        if "heading_err" in fl:
            tail = fl["heading_err"][i, max(k - int(tail_s * control_hz), 0) : k]
            row["drift"] = sustained(tail > drift_deg, int(drift_s * control_hz))
            row["heading_err_mean"] = float(fl["heading_err"][i, :k].mean())
        last = slice(max(k - control_hz, 0), k)
        row["peak_yaw_rate_last_1s"] = float(np.abs(fl["yaw_rate"][i, last]).max())
        row["speed_last_0.5s"] = float(
            np.linalg.norm(vel[max(k - control_hz // 2, 0) : k, :2], axis=1).mean()
        )
        row["approaches"] = approaches(pos, vel, obstacles, k, hit)
        rows.append(row)
    return rows


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(not r["failed"] for r in rows)
    lo, hi = wilson_interval(ok, n)
    ends = Counter(r["end"] for r in rows)
    hits = [r for r in rows if r["end"] == "obstacle"]
    failures = [r for r in rows if r["failed"]]
    out = {
        "n": n,
        "success": ok / n,
        "wilson95": [lo, hi],
        "ends": {k: ends.get(k, 0) for k in ENDS},
        "x_end_failures": np.histogram(
            [r["x_end"] for r in failures], bins=[0, 4, 8, 12, 16, 30]
        )[0].tolist(),
        "x_end_mean_all": mean(r["x_end"] for r in rows),
    }
    if "drift" in rows[0]:
        out["drift_rate"] = {
            "failures": mean(r["drift"] for r in failures),
            "successes": mean(r["drift"] for r in rows if not r["failed"]),
            "all": mean(r["drift"] for r in rows),
        }
    out["hit_rank"] = dict(
        sorted(
            Counter(
                a["rank"] for r in hits for a in r["approaches"] if a["hit"]
            ).items()
        )
    )
    out["hits"] = {
        "n": len(hits),
        "speed_last_0.5s": mean(r["speed_last_0.5s"] for r in hits),
        "peak_yaw_rate_last_1s": mean(r["peak_yaw_rate_last_1s"] for r in hits),
        "mid_turn": mean(r["peak_yaw_rate_last_1s"] > 1.0 for r in hits),
        "off_axis_45deg": mean(abs(r["rel_yaw_end_deg"]) > 45 for r in hits),
    }
    on_course = [a for r in rows for a in r["approaches"] if a["on_course"]]
    out["on_course"] = {
        "n": len(on_course),
        "cleared": mean(not a["hit"] for a in on_course),
        "onset_m": mean(a["onset_m"] for a in on_course),
        "onset_m_cleared": mean(a["onset_m"] for a in on_course if not a["hit"]),
        "onset_m_hit": mean(a["onset_m"] for a in on_course if a["hit"]),
    }
    for dist in DISTANCES:
        out["on_course"][f"clearance_{dist}"] = mean(
            a[f"clearance_{dist}"] for a in on_course
        )
        out["on_course"][f"vx_{dist}"] = mean(a[f"vx_{dist}"] for a in on_course)
    return out


def fly_runs(args) -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seed, args.seed + args.n))
    for run in map(Path, args.runs):
        out = run / FLIGHTS
        if out.exists() and not args.force:
            print(f"{out} exists, skipping")
            continue
        model, cfg = load_checkpoint(run / args.ckpt, device)
        eps = fly(
            cfg,
            seeds,
            PolicyDriver(model, cfg, args.n_envs),
            args.n_envs,
            progress=True,
        )
        np.savez_compressed(out, **pack(eps))
        print(f"{run.name}: success {np.mean([e.success for e in eps]):.3f} -> {out}")
        del model
        torch.cuda.empty_cache()
    return 0


def fly_expert(args) -> int:
    _, cfg = load_checkpoint(Path(args.like) / args.ckpt)
    seeds = list(range(args.seed, args.seed + args.n))
    eps = fly(cfg, seeds, ExpertDriver(cfg, args.n_envs), args.n_envs, progress=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.out) / FLIGHTS, **pack(eps))
    print(f"expert: success {np.mean([e.success for e in eps]):.3f}")
    return 0


def condition(run: Path) -> str:
    return re.sub(r"_s\d+$", "", run.name)


def fmt(v, spec=".2f") -> str:
    return "-" if v is None else format(v, spec)


def print_run(name: str, s: dict) -> None:
    e = s["ends"]
    lo, hi = s["wilson95"]
    drift = s.get("drift_rate", {})
    print(
        f"{name:24s} {100 * s['success']:5.1f} [{100 * lo:4.1f},{100 * hi:4.1f}]  "
        f"obst {e['obstacle']:3d} wall {e['side_wall']:3d} floor {e['floor']:2d} ceil {e['ceiling']:2d} "
        f"oob {e['out_of_bounds']:2d}  drift fail/ok {fmt(drift.get('failures'))}/{fmt(drift.get('successes'))}  "
        f"x_fail {s['x_end_failures']}"
    )


def print_approach(name: str, s: dict) -> None:
    c, h = s["on_course"], s["hits"]
    print(
        f"{name:24s} on-course {c['n']:4d} cleared {fmt(c['cleared'])}  onset {fmt(c['onset_m'])} m "
        f"(cleared {fmt(c['onset_m_cleared'])}, hit {fmt(c['onset_m_hit'])})  "
        f"clear@3/2/1/0.5 m "
        + "/".join(fmt(c[f"clearance_{d}"]) for d in (3.0, 2.0, 1.0, 0.5))
        + "  vx@3/2/1/0.5 "
        + "/".join(fmt(c[f"vx_{d}"]) for d in (3.0, 2.0, 1.0, 0.5))
        + f"  | hits {h['n']}: speed {fmt(h['speed_last_0.5s'])} mid-turn {fmt(h['mid_turn'])} "
        f"off-axis {fmt(h['off_axis_45deg'])} rank {s['hit_rank']}"
    )


def report(args) -> int:
    from flybywire.config import load_config

    cfg = load_config()
    half_width, height = float(cfg.env.corridor.width_m) / 2, float(
        cfg.env.corridor.height_m
    )
    summaries, by_condition = {}, defaultdict(list)
    sources = [("expert", Path(args.expert) / FLIGHTS)] if args.expert else []
    sources += [(Path(r).name, Path(r) / FLIGHTS) for r in args.runs]
    for name, path in sources:
        with np.load(path) as z:
            fl = {k: z[k] for k in z.files}
        rows = episode_rows(
            fl, half_width, height, args.drift_deg, args.drift_s, args.tail_s
        )
        summaries[name] = summarize_rows(rows)
        if name != "expert":
            by_condition[condition(Path(name))] += rows
    pooled = {c: summarize_rows(r) for c, r in by_condition.items()}
    print(
        "success % [Wilson 95], how the episodes ended, drift rate among failures / successes, "
        "x of failures in 0-4-8-12-16-30 m bins"
    )
    for name, s in summaries.items():
        print_run(name, s)
    for c, s in pooled.items():
        print_run(f"{c} (pooled)", s)
    print("\nobstacle approaches")
    for name, s in {
        **summaries,
        **{f"{c} (pooled)": s for c, s in pooled.items()},
    }.items():
        print_approach(name, s)
    for c in pooled:
        rates = [
            summaries[Path(r).name]["success"]
            for r in args.runs
            if condition(Path(r)) == c
        ]
        sd = np.std(rates, ddof=1) if len(rates) > 1 else float("nan")
        print(
            f"{c}: per-run success {[round(x, 3) for x in rates]} mean {np.mean(rates):.3f} sd {sd:.3f}"
        )
    if args.out:
        Path(args.out).write_text(
            json.dumps({"runs": summaries, "conditions": pooled}, indent=2)
        )
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--n", type=int, default=200)
    common.add_argument("--seed", type=int, default=10000)
    common.add_argument("--n-envs", type=int, default=12)
    common.add_argument("--ckpt", default="model_best.pt")
    f = sub.add_parser("fly", parents=[common])
    f.add_argument("runs", nargs="+")
    f.add_argument("--force", action="store_true")
    f.set_defaults(func=fly_runs)
    e = sub.add_parser("fly-expert", parents=[common])
    e.add_argument(
        "--like",
        required=True,
        help="run whose checkpoint config sets the env and expert",
    )
    e.add_argument("--out", required=True)
    e.set_defaults(func=fly_expert)
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    r.add_argument("--expert", default=None, help="directory written by fly-expert")
    r.add_argument("--out", default=None)
    r.add_argument("--drift-deg", type=float, default=30.0)
    r.add_argument("--drift-s", type=float, default=0.5)
    r.add_argument("--tail-s", type=float, default=3.0)
    r.set_defaults(func=report)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
