#!/usr/bin/env python
"""Which way does the decoded heading drift on a no-motion input, and what sets it?

Same protocol as eval_darkness.py (real T4/T5 up to t0, then a constant input for --dark-seconds),
but the drift is kept signed: positive = turning left, like positive yaw. For every run, test
episode and t0 the core is branched under constant LPTC channel inputs:

  grey      the flyvis grey steady state (what eval_darkness.py uses)
  static    the T4/T5 steady state under the episode's last real frame held still (the world stops)
  grey/static with one channel swapped: grey with channel k set to its static value, and static
            with channel k set to its grey value, for each of the 8 channels

A static yaw offset that the network integrates as a fake turn would show up as: drift rate =
yaw-rate gain x (grey yaw - zero-turn level). The yaw-rate gain and zero-turn level come from a
fit on the real cache (decoded heading rate on the yaw channel, 0.25 s windows; a second fit uses
the true yaw rate). The channel swaps test that prediction directly. Recurrent weights are also
scaled (static input) to see how sharply persistence depends on them.

Usage:
  python scripts/diagnose_darkness.py [--runs "runs/sweep_turns/*_s[012]/model_best.pt"] [--out runs/darkness_diag]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import numpy as np
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import REPO_ROOT
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.darkness import (
    grey_t4t5,
    hold_branch,
    recurrence_scaled,
    signed_stats,
    snapshots,
    static_t4t5,
    velocity_field,
    yaw_calibration,
)
from flybywire.eval.topography import wrap
from flybywire.train.bc import BCData
from flybywire.vision.flowfields import CHANNELS

SCALES = (0.0, 0.5, 0.8, 0.9, 1.1, 1.25)
SECONDS = (0.5, 1.0, 2.0)


def real_pass(model, data, t0s, chunk):
    """Channels, decoded angle, yaw rate and true heading of every test episode, plus the core state at each t0."""
    s, angle, yaw_rate, rel_yaw = [], [], [], []
    states = {t0: ([], []) for t0 in t0s}
    for start in range(0, len(data), chunk):
        idx = list(range(start, min(start + chunk, len(data))))
        batch = data.batch(idx, "cpu")
        x = batch["x16"].float()
        with torch.no_grad():
            out = model.run(x, model.init_state(len(idx)))
        h = out["heading"].numpy()
        s.append(out["s"].numpy())
        angle.append(np.arctan2(h[..., 0], h[..., 1]))
        yaw_rate.append(
            np.stack([Trajectory.load(data.cache.trajectories[i], load_hexals=False).yaw_rate for i in idx])
        )
        rel_yaw.append(torch.atan2(batch["heading"][..., 0], batch["heading"][..., 1]).numpy())
        for t0, (v, r) in snapshots(model, x, t0s).items():
            states[t0][0].append(v)
            states[t0][1].append(r)
    states = {t0: (torch.cat(v), torch.cat(r)) for t0, (v, r) in states.items()}
    return [np.concatenate(a) for a in (s, angle, yaw_rate, rel_yaw)], states


def branch_drift(model, states, t0s, s, n_steps, sel, scale=1.0):
    """Signed drift (deg, steps sel after t0), decoder size ratio and start angle; trials are t0-major.

    s: LPTC channels held during the branch, (8,) or (n_t0, B, 8).
    """
    drift, size, angle0 = [], [], []
    for i, t0 in enumerate(t0s):
        v, r = states[t0]
        with recurrence_scaled(model.core, scale):
            rs = hold_branch(model, v, r, s if s.ndim == 1 else s[i], n_steps)
        with torch.no_grad():
            head = model.heading(rs[:, sel])
            dec = (head - model.heading.bias).norm(dim=-1).numpy()
        head = head.numpy()
        ang = np.arctan2(head[..., 0], head[..., 1])
        drift.append(np.degrees(wrap(ang - ang[:, :1])))
        size.append(dec / dec[:, :1])
        angle0.append(ang[:, 0])
    return np.concatenate(drift), np.concatenate(size), np.concatenate(angle0)


def summarise(branch, start_true, episode, subsets, steps_at):
    """Direction statistics per subset and time, plus how the drift depends on the heading it starts from."""
    drift, size, start_decoded = branch
    out = {}
    for name, keep in subsets.items():
        per = {}
        for sec, k in steps_at.items():
            st = signed_stats(drift[keep, k], episode[keep])
            st["median_abs"] = float(np.median(np.abs(drift[keep, k])))
            st["decoder_size_ratio"] = float(size[keep, k].mean())
            per[f"{sec:g}s"] = st
        k = steps_at[SECONDS[-1]]
        for label, start in (("decoded", start_decoded), ("true", start_true)):
            slope, icpt = np.polyfit(np.degrees(wrap(start[keep])), drift[keep, k], 1)
            per[f"vs_{label}_start_heading_2s"] = {"slope_deg_per_deg": float(slope), "intercept_deg": float(icpt)}
        out[name] = per
    return out


def swap_table(branches, s_grey, s_static, steps_at, keep):
    """Per channel: grey - static, and the mean signed drift change from swapping only that channel."""
    k5, k20 = steps_at[0.5], steps_at[2.0]
    mean = lambda name, k: float(branches[name][0][keep, k].mean())  # noqa: E731
    rows = {}
    for i, ch in enumerate(CHANNELS):
        d_ch = float(s_grey[i] - s_static[:, i].mean())
        add = {f"{sec:g}s": mean(f"static+grey_{ch}", k) - mean("static", k) for sec, k in ((0.5, k5), (2.0, k20))}
        rem = {f"{sec:g}s": mean("grey", k) - mean(f"grey+static_{ch}", k) for sec, k in ((0.5, k5), (2.0, k20))}
        rows[ch] = {
            "grey_minus_static": round(d_ch, 3),
            "drift_deg_adding_grey_channel_to_static": add,
            "drift_deg_removing_offset_from_grey": rem,
            "dc_gain_rad_s_per_unit": float(np.radians(add["0.5s"]) / 0.5 / d_ch) if abs(d_ch) > 1e-6 else None,
        }
    return rows


def diagnose(ckpt, data, grey, frames, args):
    model, _ = load_checkpoint(ckpt, "cpu")
    t0s = [int(round(t * 100)) for t in args.t0]
    n_steps = int(round(args.dark_seconds * 100))
    sel = np.arange(0, n_steps + 1, 5)
    steps_at = {sec: int(round(sec * 100)) // 5 for sec in SECONDS}
    (s, angle, yaw_rate, rel_yaw), states = real_pass(model, data, t0s, args.chunk)
    n_ep, n_t0 = len(s), len(t0s)

    with torch.no_grad():
        s_grey = model.sensory(grey[None])[0]
        s_static = model.sensory(torch.from_numpy(frames)).reshape(n_t0, n_ep, 8)
    cal = yaw_calibration(s[..., 0], angle, yaw_rate)

    inputs = {"grey": s_grey, "static": s_static}
    for i, ch in enumerate(CHANNELS):
        to_static = s_grey.expand(n_t0, n_ep, 8).clone()
        to_static[..., i] = s_static[..., i]
        to_grey = s_static.clone()
        to_grey[..., i] = s_grey[i]
        inputs[f"grey+static_{ch}"], inputs[f"static+grey_{ch}"] = to_static, to_grey
    branches = {name: branch_drift(model, states, t0s, c, n_steps, sel) for name, c in inputs.items()}
    scaled = {lam: branch_drift(model, states, t0s, s_static, n_steps, sel, lam) for lam in SCALES}

    episode = np.tile(np.arange(n_ep), n_t0)
    start_true = np.concatenate([rel_yaw[:, t0 - 1] for t0 in t0s])
    still = np.concatenate([np.abs(yaw_rate[:, t0 - 50 : t0]).mean(1) < args.still_rate for t0 in t0s])
    off_axis = np.concatenate([np.abs(rel_yaw[:, t0 - 1]) > np.deg2rad(45) for t0 in t0s])
    subsets = {"all": np.ones(len(episode), bool), "still_off_axis": still & off_axis}
    drift = {name: summarise(branches[name], start_true, episode, subsets, steps_at) for name in ("grey", "static")}
    fields = {
        name: velocity_field(branches[name][0][:, steps_at[1.0]], branches[name][2]) for name in ("grey", "static")
    }
    scaled_stats = {}
    for lam, b in scaled.items():
        st = summarise(b, start_true, episode, subsets, steps_at)["all"]
        scaled_stats[f"{lam:g}"] = {
            "mean_abs_0.5s": st["0.5s"]["mean_abs"],
            "mean_abs_2s": st["2s"]["mean_abs"],
            "mean_signed_2s": st["2s"]["mean"],
            "decoder_size_ratio_2s": st["2s"]["decoder_size_ratio"],
        }

    yaw = s_grey[0].item()
    naive = {
        f"{sec:g}s": {
            fit: float(np.degrees(c["gain_rad_s_per_unit"] * (yaw - c["zero_turn_level"]) * sec))
            for fit, c in cal.items()
        }
        for sec in SECONDS
    }
    name = Path(ckpt).parent.name
    np.savez_compressed(
        Path(args.out) / f"{name}_drift.npz",
        t_s=sel / 100,
        episode=episode,
        still=still,
        off_axis=off_axis,
        start_decoded=branches["grey"][2],
        start_true=start_true,
        s_static=s_static.reshape(-1, 8).numpy(),
        **{f"drift_{k}": v[0].astype(np.float32) for k, v in branches.items() if k in ("grey", "static")},
    )
    return {
        "run": name,
        "channels": {
            "grey": s_grey.numpy().round(3).tolist(),
            "static_mean": s_static.reshape(-1, 8).mean(0).numpy().round(3).tolist(),
            "static_sd": s_static.reshape(-1, 8).std(0).numpy().round(3).tolist(),
            "real_mean": s.reshape(-1, 8).mean(0).round(3).tolist(),
            "real_sd": s.reshape(-1, 8).std(0).round(3).tolist(),
        },
        "yaw_calibration": cal,
        "grey_yaw_minus_zero_turn": float(yaw - cal["decoded"]["zero_turn_level"]),
        "static_yaw_minus_zero_turn": float(s_static[..., 0].mean() - cal["decoded"]["zero_turn_level"]),
        "naive_yaw_prediction_deg": naive,
        "channel_swaps": swap_table(
            branches, s_grey.numpy(), s_static.reshape(-1, 8).numpy(), steps_at, subsets["all"]
        ),
        "drift": drift,
        "velocity_field": fields,
        "recurrence_scaled_static": scaled_stats,
        "n_trials": {k: int(v.sum()) for k, v in subsets.items()},
    }


def pool(per_run: list[dict]) -> dict:
    """Across runs: how many drift the same way, per mask condition."""
    out = {}
    for cond in ("all", "real", "rewire"):
        rs = [r for r in per_run if cond == "all" or r["run"].startswith(cond)]
        row = {"n_runs": len(rs)}
        for branch in ("grey", "static"):
            for sub in ("all", "still_off_axis"):
                m = np.array([r["drift"][branch][sub]["2s"]["mean"] for r in rs])
                row[f"{branch}_{sub}"] = {
                    "runs_negative": int((m < 0).sum()),
                    "mean_of_run_means_deg": float(m.mean()),
                    "sd_of_run_means_deg": float(m.std(ddof=1)) if len(m) > 1 else 0.0,
                }
        out[cond] = row
    return out


def plot(path: Path, per_run: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = {"grey": "#eb6834", "static": "#2a78d6"}
    labels = {"grey": "grey input", "static": "last frame held still"}
    curves = {k: [] for k in colors}
    for r in per_run:
        z = np.load(out_dir / f"{r['run']}_drift.npz")
        for k in colors:
            curves[k].append(z[f"drift_{k}"].mean(0))
    t = z["t_s"]
    for k, c in colors.items():
        for y in curves[k]:
            ax[0, 0].plot(t, y, color=c, lw=0.6, alpha=0.4)
        ax[0, 0].plot(t, np.mean(curves[k], 0), color=c, lw=2, label=labels[k])
    ax[0, 0].axhline(0, color="#999999", lw=0.8)
    ax[0, 0].set_xlabel("seconds after input stops")
    ax[0, 0].set_ylabel("mean signed drift of decoded heading (deg)")
    ax[0, 0].legend(fontsize=7, frameon=False)

    obs = [r["drift"]["grey"]["all"]["2s"]["mean"] for r in per_run]
    for fit, mk in (("decoded", "o"), ("true", "s")):
        ax[0, 1].scatter(
            [r["naive_yaw_prediction_deg"]["2s"][fit] for r in per_run],
            obs,
            marker=mk,
            s=18,
            label=f"gain from {fit} rate",
        )
    lim = [min(obs) - 5, max(obs) + 5]
    ax[0, 1].plot(lim, lim, color="#999999", lw=0.8)
    ax[0, 1].set_xlabel("yaw-only prediction at 2 s (deg)")
    ax[0, 1].set_ylabel("observed mean signed drift at 2 s (deg)")
    ax[0, 1].legend(fontsize=7, frameon=False)

    for i, ch in enumerate(CHANNELS):
        v = [r["channel_swaps"][ch]["drift_deg_adding_grey_channel_to_static"]["2s"] for r in per_run]
        ax[1, 0].scatter([i] * len(v), v, color="#2a78d6", s=12)
    ax[1, 0].set_xticks(range(len(CHANNELS)), CHANNELS, rotation=45, ha="right", fontsize=7)
    ax[1, 0].set_ylabel("change in mean drift at 2 s when only\nthis channel goes grey (deg)")
    ax[1, 0].axhline(0, color="#999999", lw=0.8)

    cond_color = {"real": "#2a78d6", "rewire": "#eb6834"}
    for r in per_run:
        f = r["velocity_field"]["static"]
        ax[1, 1].plot(
            f["centre_deg"], f["median_drift_1s_deg"], color=cond_color[r["run"].split("_")[0]], lw=0.8, alpha=0.7
        )
    ax[1, 1].axhline(0, color="#999999", lw=0.8)
    ax[1, 1].set_xlabel("decoded heading at t0 (deg)")
    ax[1, 1].set_ylabel("median drift over the next 1 s, held frame (deg)")
    ax[1, 1].plot([], [], color=cond_color["real"], label="real mask")
    ax[1, 1].plot([], [], color=cond_color["rewire"], label="rewired mask")
    ax[1, 1].legend(fontsize=7, frameon=False)
    for a in ax.flat:
        a.grid(color="#dddddd", lw=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--runs", nargs="+", default=[str(REPO_ROOT / "runs" / "sweep_turns" / "*_s[012]" / "model_best.pt")]
    )
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "cache_turns_test"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "darkness_diag"))
    ap.add_argument("--t0", type=float, nargs="+", default=np.arange(1.5, 8.01, 0.25).tolist(), help="seconds")
    ap.add_argument("--dark-seconds", type=float, default=2.0)
    ap.add_argument("--still-rate", type=float, default=0.3, help="rad/s")
    ap.add_argument("--hold-seconds", type=float, default=1.0, help="how long a frame is held to get its static T4/T5")
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default=None, help="for the flyvis steps only")
    ap.add_argument("--flyvis-chunk", type=int, default=50)
    ap.add_argument("--replot", action="store_true", help="redraw the figure from the existing json and npz files")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.replot:
        plot(
            out_dir / "diagnose_darkness.png",
            json.loads((out_dir / "diagnose_darkness.json").read_text())["runs"],
            out_dir,
        )
        return 0
    ckpts = sorted(p for pat in args.runs for p in glob.glob(pat))
    cfg = load_checkpoint(ckpts[0], "cpu")[1]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = BCData(T4T5Cache(args.cache), clocks_from_cfg(cfg))
    grey = grey_t4t5(cfg, device).cpu()
    last = [int(round(t * 100)) - 1 for t in args.t0]
    hexals = np.stack([Trajectory.load(p).hexals[last] for p in data.cache.trajectories], axis=1)
    frames = static_t4t5(cfg, device, hexals.reshape(-1, 721), args.hold_seconds, args.flyvis_chunk)
    frames = frames.astype(np.float16).astype(np.float32)
    gc.collect()
    torch.cuda.empty_cache()  # the GPU is shared; the rest of this script runs on the CPU

    per_run = []
    for ckpt in ckpts:
        res = diagnose(ckpt, data, grey, frames, args)
        per_run.append(res)
        g, s = res["drift"]["grey"]["all"]["2s"], res["drift"]["static"]["all"]["2s"]
        yaw_only = res["naive_yaw_prediction_deg"]["2s"]["decoded"]
        n_ep = g["episodes_negative"] + g["episodes_positive"]
        print(
            f"{res['run']:22s} 2s: grey mean {g['mean']:6.1f} |.| {g['mean_abs']:5.1f} neg {g['frac_negative']:.2f}  "
            f"episodes neg {g['episodes_negative']}/{n_ep}  "
            f"static mean {s['mean']:6.1f} |.| {s['mean_abs']:5.1f}  yaw-only prediction {yaw_only:6.1f}",
            flush=True,
        )
    report = {"runs": per_run, "pooled": pool(per_run), "t0_seconds": args.t0, "cache": args.cache}
    (out_dir / "diagnose_darkness.json").write_text(json.dumps(report, indent=2))
    plot(out_dir / "diagnose_darkness.png", per_run, out_dir)
    print(json.dumps(report["pooled"], indent=1))
    print(f"wrote {out_dir / 'diagnose_darkness.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
