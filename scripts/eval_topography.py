#!/usr/bin/env python
"""Does each EPG neuron's preferred heading in the trained network match its anatomical angle?

The checkpoint runs over its validation episodes (from the T4/T5 cache, no flyvis). Per EPG
neuron the preferred heading phi comes from a least-squares fit of rate on [1, sin h, cos h]
(h = true relative yaw), and separately from the heading decoder's weights. Both are compared
with the anatomical angle theta from scripts/epg_anatomy.py using scores that ignore the unknown
rotation and reflection of theta; p values shuffle neuron labels.

Headings in the expert data are small, so cos h hardly varies and phi is poorly determined. This
script reports how poorly (bootstrap over episodes) next to the scores. A third score uses only
the slope on sin h, which small headings do determine: if tuning is a bump at phi, the slope is
proportional to sin(phi), so it should be a sinusoid of theta whatever the rotation.

Usage:
  python scripts/eval_topography.py [--ckpt runs/bc/real_s0/model.pt] [--cache data/cache]
                                    [--angles data/flywire/processed/epg_eb_angles.npz] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import REPO_ROOT, data_dir
from flybywire.connectome.masks import load_base_connectome
from flybywire.data.cache import T4T5Cache
from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.topography import (
    abs_circ_corr,
    alignment,
    alignment_score,
    bootstrap_tuning,
    circ_corr,
    cos_over_sin_se,
    decoder_angle,
    episode_moments,
    expected_alignment,
    fit_tuning,
    harmonic_r2,
    permutation_test,
    phi_spread,
    stack_moments,
    val_split,
    wrap,
)
from flybywire.train.bc import BCData


@torch.no_grad()
def epg_rates(model, data: BCData, idx, device, washout: int, chunk: int = 10):
    """Per-episode EPG rates (T, n_epg) and true relative yaw (T,), after the washout."""
    epg = torch.as_tensor(model.groups["EPG"], device=device)
    rates, heading = [], []
    for start in range(0, len(idx), chunk):
        sub = list(idx[start : start + chunk])
        batch = data.batch(sub, device)
        out = model.run(batch["x16"].float(), model.init_state(len(sub)), t0=0)
        r = out["r"].index_select(-1, epg).cpu().numpy()
        h = torch.atan2(batch["heading"][..., 0], batch["heading"][..., 1]).cpu().numpy()
        for j, i in enumerate(sub):
            n = data.cache.length(i)
            rates.append(r[j, washout:n])
            heading.append(h[j, washout:n])
    return rates, heading


def score_set(phi, theta, n_perm: int, boot_phi=None) -> dict:
    """Alignment and circular correlation of phi with theta, with permutation p values."""
    score, sign, offset = alignment(phi, theta)
    out = {
        "alignment": {**permutation_test(alignment_score, phi, theta, n_perm), "sign": sign,
                      "offset_deg": float(np.degrees(offset))},
        "circ_corr_abs": permutation_test(abs_circ_corr, phi, theta, n_perm),
        "circ_corr_signed": float(circ_corr(phi, theta)),
    }
    if boot_phi is not None:
        scores = np.array([alignment_score(p, theta) for p in boot_phi])
        out["alignment"]["bootstrap_95"] = [float(np.quantile(scores, 0.025)), float(np.quantile(scores, 0.975))]
        out["alignment"]["bootstrap_sd"] = float(scores.std())
    return out


def run_scores(name: str, keep: np.ndarray, theta, fit, boot, dec_w, n_perm: int) -> dict:
    t = theta[keep]
    dec_phi = decoder_angle(dec_w)
    slope_boot = np.array([harmonic_r2(s[keep], t) for s in boot["slope"]])
    slope = permutation_test(harmonic_r2, fit["slope"][keep], t, n_perm)
    slope["bootstrap_95"] = [float(np.quantile(slope_boot, 0.025)), float(np.quantile(slope_boot, 0.975))]
    return {
        "n_neurons": int(keep.sum()),
        "activity_phi": score_set(fit["phi"][keep], t, n_perm, boot["phi"][:, keep]),
        "decoder_phi": score_set(dec_phi[keep], t, n_perm),
        "activity_slope_harmonic_r2": slope,
        "decoder_sin_weight_harmonic_r2": permutation_test(harmonic_r2, dec_w[0, keep], t, n_perm),
        "activity_phi_vs_decoder_phi": permutation_test(alignment_score, fit["phi"][keep], dec_phi[keep], n_perm),
        "subset": name,
    }


def headline(r: dict) -> dict:
    scores = {"activity phi alignment": r["activity_phi"]["alignment"],
              "decoder phi alignment": r["decoder_phi"]["alignment"],
              "activity slope harmonic R2": r["activity_slope_harmonic_r2"]}
    return {k: f"{v['value']:.3f} (p {v['p']:.3f})" for k, v in scores.items()}


def plot(path: Path, theta, ok, fit, boot, half, dec_phi) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = np.degrees
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    blue, ink = "#4c78a8", "#333333"

    def style(a, xlabel, ylabel, title):
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(color="#dddddd", linewidth=0.5)
        a.set_axisbelow(True)
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        a.set_title(title, fontsize=10)

    t = theta[ok]
    for a, phi, err, title in (
        (ax[0], fit["phi"][ok], deg(half[ok]), "preferred heading from rates (bars: bootstrap 95%)"),
        (ax[1], dec_phi[ok], None, "angle of decoder weights"),
    ):
        a.errorbar(deg(t), deg(phi), yerr=err, fmt="o", ms=4, color=blue, ecolor="#b8c8dc", elinewidth=1)
        score, sign, offset = alignment(phi, t)
        grid = np.linspace(-np.pi, np.pi, 200)
        fitted = deg(wrap(sign * grid + offset))
        jump = np.abs(np.diff(fitted, prepend=fitted[0])) > 180
        a.plot(deg(grid), np.where(jump, np.nan, fitted), color=ink, lw=1, ls="--")
        style(a, "anatomical angle (deg)", "preferred heading (deg)", title)
        a.text(0.03, 0.04, f"alignment {score:.2f}", transform=a.transAxes, fontsize=9, color=ink)
        a.set_xlim(-180, 180)
        a.set_ylim(-180, 180)
    ax[2].errorbar(deg(t), fit["slope"][ok], yerr=1.96 * boot["slope"][:, ok].std(0), fmt="o", ms=4, color=blue,
                   ecolor="#b8c8dc", elinewidth=1)
    style(ax[2], "anatomical angle (deg)", "slope on sin h, in rate sd", "slope on sin h vs anatomical angle")
    ax[2].axhline(0, color=ink, lw=0.8)
    ax[2].set_xlim(-180, 180)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=str(REPO_ROOT / "runs" / "bc" / "real_s0" / "model.pt"))
    ap.add_argument("--cache", default=None)
    ap.add_argument("--angles", default=None)
    ap.add_argument("--out", default=None, help="directory for topography.json (default: the checkpoint's)")
    ap.add_argument("--washout", type=int, default=50, help="core steps dropped at each episode start")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.ckpt)
    model, cfg = load_checkpoint(ckpt, device)
    clocks = clocks_from_cfg(cfg)
    cache = T4T5Cache(args.cache or data_dir(cfg) / "cache")
    data = BCData(cache, clocks)
    val = val_split(len(data), int(cfg.seed))
    anat = np.load(args.angles or data_dir(cfg) / "flywire" / "processed" / "epg_eb_angles.npz")
    epg = model.groups["EPG"]
    if not np.array_equal(anat["neuron_ids"], load_base_connectome(cfg).neuron_ids[epg]):
        raise SystemExit("anatomical angles are not in the model's EPG order")
    theta, ok = anat["angle_rad"], anat["reliable"]

    rates, heading = epg_rates(model, data, val, device, args.washout)
    h = np.concatenate(heading)
    moments = stack_moments([episode_moments(r, x) for r, x in zip(rates, heading, strict=True)])
    fit = fit_tuning(moments)
    boot = bootstrap_tuning(moments, args.n_boot)
    half = phi_spread(fit["phi"], boot["phi"])
    slope_z = fit["slope"] / np.maximum(boot["slope"].std(0), 1e-12)
    dec_w = model.heading.linear.weight.detach().cpu().numpy()
    dec_phi = decoder_angle(dec_w)

    well = half < np.deg2rad(45)
    perfect = expected_alignment(theta[ok], half[ok] / 1.96)

    def scores(name, keep):
        return run_scores(name, keep, theta, fit, boot, dec_w, args.n_perm)

    report = {
        "checkpoint": str(ckpt),
        "val_episodes": [int(i) for i in val],
        "n_samples": int(len(h)),
        "washout_core_steps": args.washout,
        "cache": {k: cache.manifest.get(k) for k in ("total_T", "config_hash", "flyvis_model")},
        "heading_distribution": {
            "sd_deg": float(np.degrees(h.std())),
            "frac_abs_below_30deg": float((np.abs(h) < np.deg2rad(30)).mean()),
            "abs_max_deg": float(np.degrees(np.abs(h).max())),
            "cos_h_min": float(np.cos(h).min()),
            "sd_cos_h": float(np.cos(h).std()),
            "sd_sin_h": float(np.sin(h).std()),
        },
        "tuning_fit": {
            "n_epg": int(len(epg)),
            "mean_rate_median": float(np.median(fit["mean"])),
            "r2_median": float(np.median(fit["r2"])),
            "r2_p90": float(np.quantile(fit["r2"], 0.9)),
            "strength_median": float(np.median(fit["strength"])),
            "n_r2_above_0.05": int((fit["r2"] > 0.05).sum()),
        },
        "phi_reliability_bootstrap": {
            "n_boot": args.n_boot,
            "ci95_halfwidth_deg_median": float(np.degrees(np.median(half))),
            "ci95_halfwidth_deg_quartiles": np.degrees(np.quantile(half, [0.25, 0.75])).round(1).tolist(),
            "frac_halfwidth_below_45deg": float(well.mean()),
            "slope_z_frac_abs_above_2": float((np.abs(slope_z) > 2).mean()),
            "phi_near_pm90_frac": float((np.abs(np.abs(np.degrees(fit["phi"])) - 90) < 30).mean()),
            "se_cos_over_se_sin": cos_over_sin_se(moments),
            "alignment_if_phi_equalled_theta": {
                "mean": float(perfect.mean()), "p5": float(np.quantile(perfect, 0.05)),
                "note": "47 reliable neurons, theta plus the per-neuron bootstrap error of phi",
            },
        },
        "reliable_anatomy": scores("47 neurons with a reliable EB angle", ok),
        "reliable_anatomy_and_well_determined_phi": (
            scores("reliable anatomy and phi bootstrap half-width < 45 deg", ok & well) if (ok & well).sum() >= 10
            else None),
        "all_51": scores("all 51 EPG neurons", np.ones(len(epg), bool)),
        "per_neuron": {
            "phi_deg": np.degrees(fit["phi"]).round(1).tolist(),
            "phi_ci95_halfwidth_deg": np.degrees(half).round(1).tolist(),
            "decoder_phi_deg": np.degrees(dec_phi).round(1).tolist(),
            "anatomical_deg": np.degrees(theta).round(1).tolist(),
            "r2": fit["r2"].round(4).tolist(),
            "strength": fit["strength"].round(4).tolist(),
        },
    }
    out_dir = Path(args.out) if args.out else ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topography.json").write_text(json.dumps(report, indent=2))
    plot(out_dir / "topography.png", theta, ok, fit, boot, half, dec_phi)
    for k in ("heading_distribution", "tuning_fit", "phi_reliability_bootstrap"):
        print(k, json.dumps(report[k]))
    for k in ("reliable_anatomy", "reliable_anatomy_and_well_determined_phi", "all_51"):
        if report[k]:
            print(k, report[k]["n_neurons"], "neurons:", headline(report[k]))
    print(f"wrote {out_dir / 'topography.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
