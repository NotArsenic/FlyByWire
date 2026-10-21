#!/usr/bin/env python
"""Why the LPTC channels are not zero on the flyvis grey input although a grey baseline is subtracted.

Traces the response of every checkpoint's LPTC to the grey steady state: the baseline in the
direction map (which hexals and what state it was measured on), the grey relu(activity) by hexal
ring, the template weights, and the learned gains. The response is split into the hexals inside
ring 10 (normalised radius < 0.6, where the baseline was measured) and the outer rings. It is
then compared with the response to textured frames held still (frames of the test episodes) and
with the real data, and the effect of subtracting a per-hexal reference is evaluated offline.

Usage:
  python scripts/diagnose_lptc_grey.py [--runs "runs/sweep_turns/*_s[012]/model_best.pt"] [--out runs/darkness_diag]
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
from pathlib import Path

import numpy as np
import torch

from flybywire.config import REPO_ROOT
from flybywire.data.cache import T4T5Cache
from flybywire.data.trajectories import Trajectory
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.darkness import grey_t4t5, static_t4t5
from flybywire.vision.directions import baselines, load_direction_map
from flybywire.vision.flowfields import CHANNELS, hex_coords, hexal_normalized_coords

GROUPS = {"rings_0_10": (0, 10), "rings_11_12": (11, 12), "rings_13_15": (13, 15)}


def hex_ring(extent: int) -> np.ndarray:
    u, v = hex_coords(extent)
    return np.maximum.reduce([np.abs(u), np.abs(v), np.abs(u + v)])


def pool(x, w, mask=None):
    """x (D, P) minus its baseline, w (K, D, P) effective weights -> (K,), optionally over the hexals in mask."""
    if mask is not None:
        x, w = x[:, mask], w[:, :, mask]
    return np.einsum("dp,kdp->k", x, w)


def r3(a):
    return np.round(np.asarray(a, dtype=float), 3).tolist()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            str(REPO_ROOT / "runs" / "sweep_turns" / "*_s[012]" / "model_best.pt")
        ],
    )
    ap.add_argument("--cache", default=str(REPO_ROOT / "data" / "cache_turns_test"))
    ap.add_argument("--out", default=str(REPO_ROOT / "runs" / "darkness_diag"))
    ap.add_argument(
        "--static-frames", type=int, nargs="+", default=list(range(0, 1000, 100))
    )
    ap.add_argument("--converged-seconds", type=float, default=2.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--flyvis-chunk", type=int, default=50)
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = sorted(p for pat in args.runs for p in glob.glob(pat))
    models = {Path(c).parent.name: load_checkpoint(c, "cpu") for c in ckpts}
    cfg = next(iter(models.values()))[1]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    types = tuple(cfg.vision.cell_types)
    dm = load_direction_map(cfg.lptc.get("direction_map"), cell_types=types)
    b = baselines(dm, types)
    ring = hex_ring(int(cfg.vision.boxeye.extent))
    radius = np.linalg.norm(
        hexal_normalized_coords(
            int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size)
        ),
        axis=1,
    )
    central = radius < dm["stimulus"]["central_radius"]
    masks = {name: (ring >= lo) & (ring <= hi) for name, (lo, hi) in GROUPS.items()}

    relu = lambda a: np.maximum(np.asarray(a, dtype=np.float64), 0)  # noqa: E731
    grey = relu(grey_t4t5(cfg, device).cpu().numpy())
    grey_conv = relu(grey_t4t5(cfg, device, args.converged_seconds).cpu().numpy())
    data = T4T5Cache(args.cache)
    hexals = np.concatenate(
        [Trajectory.load(p).hexals[args.static_frames] for p in data.trajectories]
    )
    static = relu(static_t4t5(cfg, device, hexals, 1.0, args.flyvis_chunk))
    gc.collect()
    torch.cuda.empty_cache()  # the GPU is shared; everything below runs on the CPU
    ref_static = static.mean(0)

    rows = np.concatenate(
        [data.offsets[i] + np.arange(105, data.length(i), 10) for i in range(len(data))]
    )
    x_real = torch.from_numpy(np.asarray(data.data[rows]).astype(np.float32))
    yaw_rate = np.concatenate(
        [
            Trajectory.load(p, load_hexals=False).yaw_rate[105::10]
            for p in data.trajectories
        ]
    )

    w1 = next(iter(models.values()))[0].lptc.w.numpy().astype(np.float64)
    report = {
        "baseline_in_direction_map": dict(zip(types, b.round(5).tolist(), strict=True)),
        "baseline_measured_on": dm["stimulus"]["response"]
        + f", hexals with normalised radius < {dm['stimulus']['central_radius']}"
        f" ({int(central.sum())} of {len(central)}, rings 0-{int(ring[central].max())}), state after "
        f"{cfg.vision.steady_state_seconds} s of grey",
        "n_hexals": {name: int(m.sum()) for name, m in masks.items()},
        "grey_relu_mean_by_type": {
            "baseline": r3(b),
            "central_0.5s": r3(grey[:, central].mean(1)),
            **{f"{name}_0.5s": r3(grey[:, m].mean(1)) for name, m in masks.items()},
            "central_converged": r3(grey_conv[:, central].mean(1)),
            "all_converged": r3(grey_conv.mean(1)),
            "max_over_hexals_0.5s": r3(grey.max(1)),
        },
        "grey_relu_mean_by_ring": [
            r3(grey[:, ring == k].mean()) for k in range(int(ring.max()) + 1)
        ],
        "template_sum_over_hexals": {
            ch: r3(w1[k].sum(1)) for k, ch in enumerate(CHANNELS)
        },
        "template_squared_norm_share": {
            name: r3((w1[:, :, m] ** 2).sum((1, 2))) for name, m in masks.items()
        },
        "static_relu_mean_by_ring": [
            r3(static[:, :, ring == k].mean()) for k in range(int(ring.max()) + 1)
        ],
        "static_sd_over_frames_by_group": {
            name: r3(static[:, :, m].std(0).mean()) for name, m in masks.items()
        },
        "runs": {},
    }

    for run, (model, _) in models.items():
        lp = model.lptc
        w = lp.effective_weights().detach().numpy().astype(np.float64)
        base = lp.baseline.numpy().astype(np.float64)[:, None]

        def resp(a, ref=base, m=None, w=w):
            return pool(a - ref, w, m)

        with torch.no_grad():
            s_real = model.sensory(x_real).numpy()
        s_static = np.stack([resp(f) for f in static])
        shift_to_static = pool(
            base - ref_static, w
        )  # what a static reference adds to every channel
        by_group = {}
        for name, m in masks.items():
            real = np.einsum(
                "ndp,kdp->nk", x_real.numpy()[:, :, m].clip(min=0), w[:, :, m]
            ) - pool(base * np.ones(721), w, m)
            by_group[name] = {
                "grey": r3(resp(grey, m=m)),
                "static_mean": r3(np.stack([resp(f, m=m) for f in static]).mean(0)),
                "real_mean": r3(real.mean(0)),
                "real_sd": r3(real.std(0)),
            }
        report["runs"][run] = {
            "gain_mean_min_max": r3(
                [lp.g.mean().item(), lp.g.min().item(), lp.g.max().item()]
            ),
            "grey": {
                "total": r3(resp(grey)),
                **{name: r3(resp(grey, m=m)) for name, m in masks.items()},
            },
            "grey_converged": r3(resp(grey_conv)),
            "static_mean": r3(s_static.mean(0)),
            "static_sd": r3(s_static.std(0)),
            "real_mean": r3(s_real.mean(0)),
            "real_sd": r3(s_real.std(0)),
            "grey_minus_static": r3(resp(grey) - s_static.mean(0)),
            "reference_static": {
                "shift_added_to_every_sample": r3(shift_to_static),
                "grey": r3(resp(grey, ref_static)),
                "static_mean": r3(s_static.mean(0) + shift_to_static),
                "real_mean": r3(s_real.mean(0) + shift_to_static),
                "residual_sd_over_frames": r3(
                    np.stack([resp(f, ref_static) for f in static]).std(0)
                ),
            },
            "by_ring_group": by_group,
            "reference_grey": {
                "static_mean": r3(s_static.mean(0) - resp(grey)),
                "real_mean": r3(s_real.mean(0) - resp(grey)),
            },
        }
        if run == "real_learned_s0":
            parts = {
                name: np.einsum(
                    "ndp,kdp->nk", x_real.numpy()[:, :, m].clip(min=0), w[:, :, m]
                )
                for name, m in masks.items()
            }
            report["edge_information_real_learned_s0"] = {
                "channel_var_share": {
                    name: r3(p.var(0) / s_real.var(0)) for name, p in parts.items()
                },
                "yaw_channel_corr_with_yaw_rate": {
                    name: float(np.corrcoef(p[:, 0], yaw_rate)[0, 1])
                    for name, p in parts.items()
                }
                | {
                    "total": float(np.corrcoef(s_real[:, 0], yaw_rate)[0, 1]),
                    "rings_0_12": float(
                        np.corrcoef(
                            parts["rings_0_10"][:, 0] + parts["rings_11_12"][:, 0],
                            yaw_rate,
                        )[0, 1]
                    ),
                },
            }
        outer = resp(grey, m=masks["rings_13_15"]) / resp(grey)
        print(
            f"{run:20s} grey {np.round(resp(grey), 2)}  outer rings share {np.round(outer, 2)}"
        )

    (out_dir / "diagnose_lptc_grey.json").write_text(json.dumps(report, indent=2))
    plot(out_dir / "diagnose_lptc_grey.png", report, grey, ring, types)
    print(f"wrote {out_dir / 'diagnose_lptc_grey.json'}")
    return 0


def plot(path: Path, report: dict, grey, ring, types) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [
        "#2a78d6",
        "#eb6834",
        "#1baf7a",
        "#eda100",
        "#e87ba4",
        "#008300",
        "#6250d6",
        "#e34948",
    ]
    rings = np.arange(int(ring.max()) + 1)
    for i, t in enumerate(types):
        ax[0].plot(
            rings,
            [grey[i, ring == k].mean() for k in rings],
            label=t,
            lw=1.5,
            color=series[i],
        )
    for i, v in enumerate(report["baseline_in_direction_map"].values()):
        ax[0].plot([0, 10], [v, v], color=series[i], lw=0.8, ls=":")
    ax[0].axvline(10.5, color="#999999", ls="--", lw=0.8)
    ax[0].set_yscale("symlog", linthresh=0.01)
    ax[0].set_xlabel("hexal ring (0 = centre)")
    ax[0].set_ylabel("mean relu(activity) on grey (dotted: baseline)")
    ax[0].legend(fontsize=6, ncol=2, frameon=False)

    share = report["template_squared_norm_share"]
    width = 0.12
    for k, ch in enumerate(CHANNELS):
        ax[1].bar(
            np.arange(3) + (k - 3.5) * width,
            [share[g][k] for g in share],
            width,
            label=ch,
            color=series[k],
        )
    ax[1].set_xticks(
        range(3), [g.replace("rings_", "rings ").replace("_", "-") for g in share]
    )
    ax[1].set_ylabel("share of squared template norm")
    ax[1].legend(fontsize=6, ncol=2, frameon=False)

    runs = report["runs"]
    for g, color in zip(
        ("rings_0_10", "rings_11_12", "rings_13_15"), series[:3], strict=True
    ):
        ax[2].scatter(
            np.tile(np.arange(8), len(runs))
            + {"rings_0_10": -0.2, "rings_11_12": 0, "rings_13_15": 0.2}[g],
            [v for r in runs.values() for v in r["grey"][g]],
            s=8,
            color=color,
            label=g.replace("rings_", "rings ").replace("_", "-"),
        )
    ax[2].set_xticks(range(8), CHANNELS, rotation=45, ha="right", fontsize=7)
    ax[2].set_ylabel("LPTC response to grey (all runs)")
    ax[2].axhline(0, color="#999999", lw=0.8)
    ax[2].legend(fontsize=7, frameon=False)
    for a in ax:
        a.grid(color="#dddddd", lw=0.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
