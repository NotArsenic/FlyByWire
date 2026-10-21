#!/usr/bin/env python
"""Anatomical angle of every EPG neuron around the ellipsoid body (EB) from FlyWire skeletons.

The EB is found as the region where ER and EPG arbors overlap: the connected blob of smoothed ER
node density (above --frac of its peak) that contains the joint ER/EPG density peak. ER dendrites
sit in the bulbs and EPG has arbors in the bridge and gall, so neither cloud alone isolates the EB.
The ring plane's normal is the smallest-variance PCA axis of the EPG nodes inside the blob and the
centre is the blob centroid. In-plane axes (dorsal, lateral) come from the anatomy, since a ring has
no stable principal axes in its plane, and are scaled by the nodes' sd so an elliptical ring becomes
a circle. A neuron's angle is the angle of the centroid of its EB nodes; zero points dorsal and
positive turns toward +x.

The skeleton file is downloaded data: pass its path and run with `python -I`.

Usage:
  python -I scripts/epg_anatomy.py SKELETONS.parquet [--out data/flywire/processed/epg_eb_angles.npz]
                                   [--report-dir runs/anatomy] [--frac 0.3] [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import ndimage, stats

from flybywire.config import REPO_ROOT, data_dir, load_config
from flybywire.eval.topography import ring_angles, ring_frame, wrap

VOXEL_UM = 3.0
SIGMA_VOXELS = 1.5
N_WEDGES = 16
MIN_NODES = 200  # EB nodes; the four EPGt neurons barely enter the EB and fall below
MIN_RESULTANT = 0.8  # node angles must cluster, i.e. a wedge-like arbor


def load_nodes(path: str, neuron_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Root id and (x, y, z) in micrometres of every skeleton node of the given neurons."""
    f = pq.ParquetFile(path)
    want = pa.array(neuron_ids)
    root, xyz = [], []
    for g in range(f.num_row_groups):
        tab = f.read_row_group(g, columns=["neuron", "x", "y", "z"])
        tab = tab.filter(pc.is_in(tab.column("neuron"), value_set=want))
        root.append(tab.column("neuron").to_numpy())
        xyz.append(np.stack([tab.column(c).to_numpy() for c in "xyz"], axis=1))
    return np.concatenate(root), np.concatenate(xyz).astype(np.float64) / 1000.0


class Grid:
    def __init__(self, points: np.ndarray):
        lo = points.min(0)
        self.edges = [
            np.arange(lo[i], points[:, i].max() + VOXEL_UM, VOXEL_UM) for i in range(3)
        ]

    def density(self, points: np.ndarray) -> np.ndarray:
        h, _ = np.histogramdd(points, bins=self.edges)
        return ndimage.gaussian_filter(h, SIGMA_VOXELS)

    def index(self, points: np.ndarray) -> tuple:
        return tuple(
            np.clip(
                ((points[:, i] - self.edges[i][0]) // VOXEL_UM).astype(int),
                0,
                len(self.edges[i]) - 2,
            )
            for i in range(3)
        )

    def centers(self, mask: np.ndarray) -> np.ndarray:
        idx = np.argwhere(mask)
        return np.stack(
            [self.edges[i][0] + VOXEL_UM * (idx[:, i] + 0.5) for i in range(3)], axis=1
        )


def eb_mask(grid: Grid, er: np.ndarray, epg: np.ndarray, frac: float) -> np.ndarray:
    d_er, d_epg = grid.density(er), grid.density(epg)
    seed = np.unravel_index(np.argmax(d_er * d_epg), d_er.shape)
    labels, _ = ndimage.label(d_er > frac * d_er.max())
    return labels == labels[seed]


def eb_angles(
    neuron_pts: list[np.ndarray], eb_points: np.ndarray, center: np.ndarray
) -> dict:
    axes, sd = ring_frame(eb_points, center)
    centroids = np.stack([p.mean(0) for p in neuron_pts])
    node_angles = [ring_angles(p, center, axes, sd) for p in neuron_pts]
    return {
        "axes": axes,
        "sd": sd,
        "centroid_um": centroids,
        "angle": ring_angles(centroids, center, axes, sd),
        "angle_raw": np.arctan2(
            (centroids - center) @ axes[0], (centroids - center) @ axes[1]
        ),
        "resultant": np.array([np.abs(np.exp(1j * a).mean()) for a in node_angles]),
    }


def partner_jaccard(
    M: np.ndarray, epg: np.ndarray, partners: np.ndarray, direction: str
) -> np.ndarray:
    """Jaccard similarity of the partner sets of every EPG pair (partners: neuron indices)."""
    out_edges, in_edges = M[np.ix_(epg, partners)], M[np.ix_(partners, epg)].T
    sets = {"out": out_edges, "in": in_edges, "both": out_edges | in_edges}[
        direction
    ].astype(float)
    inter = sets @ sets.T
    union = sets.sum(1)[:, None] + sets.sum(1)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def pair_values(theta: np.ndarray, jac: np.ndarray):
    iu = np.triu_indices(len(theta), 1)
    return np.abs(wrap(theta[:, None] - theta[None, :]))[iu], jac[iu]


def jaccard_check(theta: np.ndarray, jac: np.ndarray, n_perm: int, seed: int) -> dict:
    """Spearman correlation of angular distance with partner Jaccard, shuffling neuron labels."""
    rng = np.random.default_rng(seed)
    dist, j = pair_values(theta, jac)
    rho = stats.spearmanr(dist, j).statistic
    rj = stats.rankdata(j)
    rj = (rj - rj.mean()) / rj.std()

    def rho_of(t):
        d, _ = pair_values(t, jac)
        rd = stats.rankdata(d)
        return float(((rd - rd.mean()) / rd.std() * rj).mean())

    null = np.array([rho_of(rng.permutation(theta)) for _ in range(n_perm)])
    near, far = dist <= np.deg2rad(30), dist >= np.deg2rad(120)
    return {
        "spearman_rho": float(rho),
        "p_one_sided_negative": float((1 + (null <= rho).sum()) / (1 + n_perm)),
        "p_two_sided": float((1 + (np.abs(null) >= abs(rho)).sum()) / (1 + n_perm)),
        "null_rho_sd": float(null.std()),
        "mean_jaccard_within_30deg": float(j[near].mean()),
        "mean_jaccard_beyond_120deg": float(j[far].mean()),
        "n_pairs": int(len(j)),
    }


def uniformity(theta: np.ndarray) -> dict:
    ang = np.sort(np.mod(theta, 2 * np.pi))
    counts = np.histogram(ang, bins=N_WEDGES, range=(0, 2 * np.pi))[0]
    n = len(ang)
    chi2 = float(((counts - n / N_WEDGES) ** 2 / (n / N_WEDGES)).sum())
    z = np.exp(1j * ang).mean()
    rn = n * abs(z)
    gaps = np.diff(np.append(ang, ang[0] + 2 * np.pi))
    return {
        "wedge_counts": counts.tolist(),
        "empty_wedges": int((counts == 0).sum()),
        "chi2_16": chi2,
        "chi2_p": float(stats.chi2.sf(chi2, N_WEDGES - 1)),
        "rayleigh_R": float(abs(z)),
        "rayleigh_p": float(
            np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - rn**2)) - (1 + 2 * n))
        ),
        "max_gap_deg": float(np.degrees(gaps.max())),
        "expected_max_gap_deg_if_uniform": float(
            np.degrees(2 * np.pi / n * (1 / np.arange(1, n + 1)).sum())
        ),
    }


def plot(path: Path, ctx: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta, ok, jac, uni, ck = (
        ctx["theta"],
        ctx["ok"],
        ctx["jaccard"],
        ctx["uniformity"],
        ctx["jaccard_check"],
    )
    axes, sd, center = ctx["axes"], ctx["sd"], ctx["center"]
    keep = np.flatnonzero(ok)
    colors = plt.get_cmap("twilight_shifted")((theta % (2 * np.pi)) / (2 * np.pi))
    colors[~ok] = (0.6, 0.6, 0.6, 1.0)
    ink, blue, red = "#333333", "#4c78a8", "#e45756"
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))

    def style(a):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(color="#dddddd", linewidth=0.5)
        a.set_axisbelow(True)

    cen = (ctx["centroids"] - center) @ axes.T
    rng = np.random.default_rng(0)
    for a, (u, v), names in (
        (ax[0, 0], (0, 1), ("lateral", "dorsal")),
        (ax[0, 1], (0, 2), ("lateral", "normal")),
    ):
        for i in keep:
            d = (ctx["neuron_pts"][i] - center) @ axes.T
            pick = rng.choice(len(d), min(len(d), 400), replace=False)
            a.scatter(
                d[pick, u], d[pick, v], s=1.5, color=colors[i], alpha=0.5, linewidths=0
            )
        a.scatter(
            cen[ok, u],
            cen[ok, v],
            s=22,
            color=colors[ok],
            edgecolor=ink,
            linewidth=0.6,
            zorder=3,
        )
        a.scatter(
            cen[~ok, u], cen[~ok, v], s=30, color=colors[~ok], marker="x", zorder=3
        )
        a.set_aspect("equal")
        a.set_xlabel(f"{names[0]} (um)")
        a.set_ylabel(f"{names[1]} (um)")
        style(a)
    ax[0, 0].set_title(
        "EB nodes of EPG neurons (dots: centroids, x: unreliable)", fontsize=10
    )
    ax[0, 1].set_title("same, edge-on", fontsize=10)

    ax[0, 2].remove()
    polar = fig.add_subplot(2, 3, 3, projection="polar")
    radius = np.hypot(*(cen[:, :2] / sd).T)
    polar.scatter(
        theta[ok],
        radius[ok],
        s=30,
        color=colors[ok],
        edgecolor=ink,
        linewidth=0.5,
        zorder=3,
    )
    polar.set_theta_zero_location("N")
    polar.set_theta_direction(-1)
    polar.set_yticks([])
    polar.set_title("centroid angle (0 = dorsal, + toward +x)", fontsize=10)

    n = len(keep)
    ax[1, 0].bar(np.arange(N_WEDGES), uni["wedge_counts"], color=blue, width=0.8)
    ax[1, 0].axhline(n / N_WEDGES, color=ink, linewidth=1, linestyle="--")
    ax[1, 0].set_xlabel("wedge (22.5 deg each)")
    ax[1, 0].set_ylabel("EPG neurons")
    ax[1, 0].set_title(
        f"angle histogram (dashed: uniform), chi2 p = {uni['chi2_p']:.2f}", fontsize=10
    )
    style(ax[1, 0])

    dist, j = pair_values(theta[keep], jac[np.ix_(keep, keep)])
    deg = np.degrees(dist)
    ax[1, 1].scatter(deg, j, s=3, color=blue, alpha=0.25, linewidths=0)
    bins = np.linspace(0, 180, 10)
    means = [
        j[(deg >= lo) & (deg < hi)].mean()
        for lo, hi in zip(bins[:-1], bins[1:], strict=True)
    ]
    ax[1, 1].plot(
        0.5 * (bins[1:] + bins[:-1]),
        means,
        color=red,
        linewidth=2,
        marker="o",
        markersize=5,
    )
    ax[1, 1].set_title(
        f"partner overlap vs angle: rho = {ck['spearman_rho']:.2f}, "
        f"p = {ck['p_one_sided_negative']:.4f}",
        fontsize=10,
    )
    ax[1, 1].set_xlabel("angular distance between two EPG neurons (deg)")
    ax[1, 1].set_ylabel("Jaccard of PEN/PEG/D7 partners")
    style(ax[1, 1])

    ax[1, 2].scatter(ctx["n_nodes"][ok], ctx["resultant"][ok], s=18, color=blue)
    ax[1, 2].scatter(
        ctx["n_nodes"][~ok], ctx["resultant"][~ok], s=24, color="#999999", marker="x"
    )
    ax[1, 2].axvline(MIN_NODES, color=ink, linewidth=1, linestyle="--")
    ax[1, 2].axhline(MIN_RESULTANT, color=ink, linewidth=1, linestyle="--")
    ax[1, 2].set_xscale("log")
    ax[1, 2].set_xlabel("EB nodes per neuron")
    ax[1, 2].set_ylabel("resultant length of node angles")
    ax[1, 2].set_title("reliability cut", fontsize=10)
    style(ax[1, 2])
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "skeletons", help="skeleton parquet (columns neuron, x, y, z in nm)"
    )
    ap.add_argument("--graph", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument(
        "--frac",
        type=float,
        default=0.3,
        help="ER density fraction that bounds the EB blob",
    )
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config()
    graph = (
        Path(args.graph)
        if args.graph
        else data_dir(cfg) / "flywire" / "processed" / "cx_graph.npz"
    )
    out = (
        Path(args.out)
        if args.out
        else data_dir(cfg) / "flywire" / "processed" / "epg_eb_angles.npz"
    )
    report_dir = (
        Path(args.report_dir) if args.report_dir else REPO_ROOT / "runs" / "anatomy"
    )
    g = np.load(graph, allow_pickle=False)
    ids, M = g["neuron_ids"], g["M"]
    epg_idx = g["group__EPG"]
    root, xyz = load_nodes(args.skeletons, ids)
    missing = set(ids) - set(root.tolist())
    print(f"{len(root)} nodes of {len(ids) - len(missing)}/{len(ids)} neurons")

    er = xyz[np.isin(root, ids[g["group__ER"]])]
    epg = xyz[np.isin(root, ids[epg_idx])]
    arbors = [xyz[root == ids[i]] for i in epg_idx]
    grid = Grid(xyz)

    def angles_for(frac: float, only: np.ndarray) -> dict:
        mask = eb_mask(grid, er, epg, frac)
        epg_in = epg[mask[grid.index(epg)]]
        center = grid.centers(mask).mean(0)
        pts = [arbors[i][mask[grid.index(arbors[i])]] for i in only]
        res = eb_angles(pts, epg_in, center)
        res.update(
            center=center,
            neuron_pts=pts,
            n_nodes=np.array([len(p) for p in pts]),
            volume_um3=float(mask.sum() * VOXEL_UM**3),
            frac_epg_in_eb=float(len(epg_in) / len(epg)),
        )
        return res

    everyone = np.arange(len(epg_idx))
    main_res = angles_for(args.frac, everyone)
    theta = main_res["angle"]
    center, axes, sd = main_res["center"], main_res["axes"], main_res["sd"]
    nn_ = main_res["n_nodes"]
    ok = (nn_ >= MIN_NODES) & (main_res["resultant"] >= MIN_RESULTANT)
    keep = np.flatnonzero(ok)
    print(
        f"{ok.sum()}/{len(ok)} EPG neurons with a reliable EB angle; dropped types: "
        f"{sorted({str(g['types'][epg_idx[i]]) for i in np.flatnonzero(~ok)})}"
    )

    def rms_deg(a, b) -> float:
        return float(np.degrees(np.sqrt((wrap(a - b) ** 2).mean())))

    sens = {}
    for frac in (0.4, 0.5):
        other = angles_for(frac, keep)
        diff = wrap(other["angle"] - theta[keep])
        sens[f"frac_{frac}"] = {
            "rms_angle_diff_deg": rms_deg(other["angle"], theta[keep]),
            "max_angle_diff_deg": float(np.degrees(np.abs(diff).max())),
            "volume_um3": other["volume_um3"],
        }
    epg_median = np.median(np.concatenate(main_res["neuron_pts"]), axis=0)
    sens["center_at_epg_node_median"] = {
        "offset_um": float(np.linalg.norm(epg_median - center)),
        "rms_angle_diff_deg": rms_deg(
            ring_angles(main_res["centroid_um"][keep], epg_median, axes, sd),
            theta[keep],
        ),
    }
    sens["no_in_plane_whitening"] = {
        "rms_angle_diff_deg": rms_deg(main_res["angle_raw"][keep], theta[keep])
    }

    partners = np.concatenate([g["group__PEN"], g["group__PEG"], g["group__D7"]])
    jac = {d: partner_jaccard(M, epg_idx, partners, d) for d in ("both", "out", "in")}
    checks = {
        d: jaccard_check(
            theta[keep], jac[d][np.ix_(keep, keep)], args.n_perm, args.seed
        )
        for d in jac
    }
    checks["both_all_51"] = jaccard_check(theta, jac["both"], args.n_perm, args.seed)
    uni = uniformity(theta[keep])

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        neuron_ids=ids[epg_idx],
        angle_rad=theta,
        centroid_nm=main_res["centroid_um"] * 1000.0,
        eb_center_nm=center * 1000.0,
        eb_normal=axes[2],
        n_eb_nodes=nn_,
        reliable=ok,
        eb_axes=axes,
        eb_sd_um=sd,
        node_angle_resultant=main_res["resultant"],
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "skeletons": str(args.skeletons),
        "eb": {
            "frac": args.frac,
            "volume_um3": main_res["volume_um3"],
            "center_um": center.round(2).tolist(),
            "normal": axes[2].round(4).tolist(),
            "normal_tilt_from_z_deg": float(np.degrees(np.arccos(abs(axes[2][2])))),
            "in_plane_sd_um": sd.round(2).tolist(),
            "fraction_of_epg_nodes_in_eb": main_res["frac_epg_in_eb"],
        },
        "n_reliable": int(ok.sum()),
        "unreliable": [
            {
                "type": str(g["types"][epg_idx[i]]),
                "instance": str(g["instances"][epg_idx[i]]),
                "n_eb_nodes": int(nn_[i]),
                "node_angle_resultant": float(main_res["resultant"][i]),
            }
            for i in np.flatnonzero(~ok)
        ],
        "n_eb_nodes_reliable": {
            "min": int(nn_[keep].min()),
            "median": float(np.median(nn_[keep])),
            "max": int(nn_[keep].max()),
        },
        "node_angle_resultant_reliable": {
            "min": float(main_res["resultant"][keep].min()),
            "median": float(np.median(main_res["resultant"][keep])),
        },
        "angle_uniformity_reliable": uni,
        "partner_jaccard_vs_angle_reliable": checks,
        "sensitivity_reliable": sens,
        "angle_convention": "0 = dorsal in the ring plane, positive toward +x (FlyWire x); arbitrary up to "
        "rotation and reflection, which the topography scores are invariant to",
    }
    (report_dir / "epg_eb_angles.json").write_text(json.dumps(report, indent=2))
    plot(
        report_dir / "epg_eb_angles.png",
        {
            "theta": theta,
            "ok": ok,
            "jaccard": jac["both"],
            "axes": axes,
            "sd": sd,
            "center": center,
            "neuron_pts": main_res["neuron_pts"],
            "centroids": main_res["centroid_um"],
            "uniformity": uni,
            "jaccard_check": checks["both"],
            "n_nodes": nn_,
            "resultant": main_res["resultant"],
        },
    )
    print(json.dumps(report, indent=2))
    print(
        f"wrote {out}, {report_dir / 'epg_eb_angles.json'}, {report_dir / 'epg_eb_angles.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
