#!/usr/bin/env python
"""Render a 3D rotating GIF and MP4 of the two FlyWire circuits used by Fly-By-Wire.

Left panel: whole adult Drosophila brain (FlyWire release 783) showing the optic-lobe motion
pathway (flyvis 65 cell types: retinotopic R1-R8 / L1-L5 / Mi / Tm layers and T4a-d / T5a-d
direction-selective outputs) together with the central complex at the brain midline.
Right panel: zoomed 3D skeletons of all 457 central-complex neurons (17,041 directed synapses)
colored by cell class (ER, EPG, PEN, PEG, Delta7, PFL3).

Usage:
  python scripts/render_brain_3d.py [--out docs/media/fly_brain_3d.gif] [--frames 90]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from flybywire.config import REPO_ROOT, data_dir, load_config

T4T5_TYPES = {"T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"}
OL_UPSTREAM_TYPES = {
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R1-6",
    "L1", "L2", "L3", "L4", "L5", "Lawf1", "Lawf2", "Am", "C2", "C3", "CT1",
    "Mi1", "Mi2", "Mi3", "Mi4", "Mi9", "Mi10", "Mi11", "Mi12", "Mi13", "Mi14", "Mi15",
    "T1", "T2", "T2a", "T3",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm5Y", "Tm5a", "Tm5b", "Tm5c", "Tm9",
    "Tm16", "Tm20", "Tm28", "Tm30",
    "TmY3", "TmY4", "TmY5a", "TmY9", "TmY10", "TmY13", "TmY14", "TmY15", "TmY18",
}

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def sample_pts(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[rng.choice(len(pts), n, replace=False)]


def load_clouds(skel_path: Path, ann_path: Path, cx_path: Path, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ann = pd.read_csv(ann_path)
    cx = np.load(cx_path, allow_pickle=False)
    cx_ids = cx["neuron_ids"]

    t4t5_ids = pa.array(ann[ann["primary_type"].isin(T4T5_TYPES)]["root_id"].to_numpy(np.int64))
    ol_ids = pa.array(ann[ann["primary_type"].isin(OL_UPSTREAM_TYPES)]["root_id"].to_numpy(np.int64))
    cx_pa = pa.array(cx_ids)

    pf = pq.ParquetFile(skel_path)
    bg_pts, ol_pts, t4t5_pts = [], [], []
    cx_root, cx_xyz = [], []

    for g in range(pf.num_row_groups):
        tab = pf.read_row_group(g, columns=["neuron", "x", "y", "z"])
        sub_bg = tab.take(pa.array(np.arange(0, len(tab), 900)))
        bg_pts.append(np.stack([sub_bg.column(c).to_numpy() for c in "xyz"], axis=1) / 1000.0)

        sub_ol = tab.take(pa.array(np.arange(0, len(tab), 220)))
        sub_ol = sub_ol.filter(pc.is_in(sub_ol.column("neuron"), value_set=ol_ids))
        ol_pts.append(np.stack([sub_ol.column(c).to_numpy() for c in "xyz"], axis=1) / 1000.0)

        sub_t4 = tab.take(pa.array(np.arange(0, len(tab), 70)))
        sub_t4 = sub_t4.filter(pc.is_in(sub_t4.column("neuron"), value_set=t4t5_ids))
        t4t5_pts.append(np.stack([sub_t4.column(c).to_numpy() for c in "xyz"], axis=1) / 1000.0)

        sub_cx = tab.filter(pc.is_in(tab.column("neuron"), value_set=cx_pa))
        cx_root.append(sub_cx.column("neuron").to_numpy()[::5])
        cx_xyz.append(np.stack([sub_cx.column(c).to_numpy()[::5] for c in "xyz"], axis=1) / 1000.0)

    cx_root = np.concatenate(cx_root)
    cx_xyz = np.concatenate(cx_xyz)
    cx_groups = {}
    for name in ("ER", "EPG", "PEN", "PEG", "D7", "PFL3"):
        g_ids = cx_ids[cx[f"group__{name}"]]
        pts = cx_xyz[np.isin(cx_root, g_ids)]
        cx_groups[name] = sample_pts(pts, 24000, rng)

    return {
        "bg": sample_pts(np.concatenate(bg_pts), 60000, rng),
        "ol": sample_pts(np.concatenate(ol_pts), 48000, rng),
        "t4t5": sample_pts(np.concatenate(t4t5_pts), 40000, rng),
        "cx": cx_groups,
    }


def rot_matrix(azim: float, elev: float) -> np.ndarray:
    ca, sa = np.cos(azim), np.sin(azim)
    ce, se = np.cos(elev), np.sin(elev)
    ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    return rx @ ry


def project_point(pt: np.ndarray, center: np.ndarray, R: np.ndarray, w: int, h: int,
                  scale: float, y_frac: float = 0.50) -> tuple[int, int, float]:
    p = (pt - center) @ R.T
    persp = float(1.0 / np.clip(1.0 + p[2] / 1800.0, 0.6, 1.6))
    u = int(round(w * 0.5 + p[0] * scale * persp))
    v = int(round(h * y_frac + p[1] * scale * persp))
    return u, v, float(p[2])


def splat_panel(w: int, h: int, layers: list[tuple[np.ndarray, tuple[float, float, float], float, int]],
                center: np.ndarray, scale: float, azim: float, elev: float,
                y_frac: float = 0.50, bg_color=(13.0, 17.0, 23.0)) -> tuple[np.ndarray, np.ndarray]:
    W, H = w * 2, h * 2
    canvas = np.full((H, W, 3), bg_color, dtype=np.float32)
    R = rot_matrix(azim, elev)

    all_xyz, all_rgb, all_alpha, all_rad = [], [], [], []
    for pts, rgb, alpha, rad in layers:
        p = (pts - center) @ R.T
        all_xyz.append(p)
        c = np.empty((len(pts), 3), dtype=np.float32)
        c[:] = rgb
        all_rgb.append(c)
        all_alpha.append(np.full(len(pts), alpha, dtype=np.float32))
        all_rad.append(np.full(len(pts), rad, dtype=np.int32))

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    alpha = np.concatenate(all_alpha, axis=0)
    rad = np.concatenate(all_rad, axis=0)

    order = np.argsort(-xyz[:, 2])
    xyz, rgb, alpha, rad = xyz[order], rgb[order], alpha[order], rad[order]

    z_std = max(float(np.std(xyz[:, 2])), 1.0)
    depth_shade = np.clip(1.0 - 0.26 * (xyz[:, 2] / (2.4 * z_std)), 0.58, 1.24)[:, None]
    rgb = np.clip(rgb * depth_shade, 0.0, 255.0)

    persp = 1.0 / np.clip(1.0 + xyz[:, 2] / 1800.0, 0.6, 1.6)
    u = np.rint(W * 0.5 + xyz[:, 0] * scale * 2.0 * persp).astype(np.int32)
    v = np.rint(H * y_frac + xyz[:, 1] * scale * 2.0 * persp).astype(np.int32)

    valid = (u >= 2) & (u < W - 2) & (v >= 2) & (v < H - 2)
    u, v, rgb, alpha, rad = u[valid], v[valid], rgb[valid], alpha[valid], rad[valid]

    offsets_1 = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets_2 = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)]

    chunk = 8192
    for start in range(0, len(u), chunk):
        sl = slice(start, start + chunk)
        uc, vc, rgbc, ac, rc = u[sl], v[sl], rgb[sl], alpha[sl, None], rad[sl]
        canvas[vc, uc] = (1.0 - ac) * canvas[vc, uc] + ac * rgbc
        m1 = rc >= 1
        if m1.any():
            u1, v1, rgb1, a1 = uc[m1], vc[m1], rgbc[m1], ac[m1] * 0.55
            for du, dv in offsets_1[1:]:
                canvas[v1 + dv, u1 + du] = (1.0 - a1) * canvas[v1 + dv, u1 + du] + a1 * rgb1
        m2 = rc >= 2
        if m2.any():
            u2, v2, rgb2, a2 = uc[m2], vc[m2], rgbc[m2], ac[m2] * 0.30
            for du, dv in offsets_2:
                canvas[v2 + dv, u2 + du] = (1.0 - a2) * canvas[v2 + dv, u2 + du] + a2 * rgb2

    out = cv2.resize(np.clip(canvas, 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
    return out, R


def draw_triad(draw: ImageDraw.ImageDraw, R: np.ndarray, cx: int, cy: int, length: int, font):
    axes = [
        (np.array([1.0, 0.0, 0.0]), "+X", (248, 113, 113)),
        (np.array([0.0, -1.0, 0.0]), "-Y", (74, 222, 128)),
        (np.array([0.0, 0.0, 1.0]), "+Z", (56, 189, 248)),
    ]
    for vec, label, col in axes:
        p = vec @ R.T
        ex = int(round(cx + p[0] * length))
        ey = int(round(cy + p[1] * length))
        draw.line([(cx, cy), (ex, ey)], fill=col, width=2)
        tx = int(round(cx + p[0] * (length + 13)))
        ty = int(round(cy + p[1] * (length + 13)))
        draw.text((tx - 8, ty - 6), label, fill=col, font=font)


def draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, col: tuple[int, int, int], font):
    tw = int(round(draw.textlength(text, font=font)))
    draw.rectangle([(x + 5, y - 9), (x + tw + 13, y + 7)], fill=(13, 17, 23), outline=col, width=1)
    draw.ellipse([(x - 3, y - 3), (x + 3, y + 3)], fill=col)
    draw.text((x + 9, y - 7), text, fill=col, font=font)


def render_frame(clouds: dict, azim: float, elev: float, width: int = 1200, height: int = 600) -> np.ndarray:
    pw = width // 2
    ph = height - 104  # top header 52 px, bottom legend 52 px

    wb_center = np.array([505.0, 248.0, 145.0])
    cx_center = np.array([526.0, 176.0, 128.0])

    cx_g = clouds["cx"]
    wb_layers = [
        (clouds["bg"], (110.0, 128.0, 158.0), 0.22, 1),
        (clouds["ol"], (56.0, 189.0, 248.0), 0.30, 1),
        (clouds["t4t5"], (251.0, 191.0, 36.0), 0.45, 1),
        (cx_g["ER"][::2], (192.0, 132.0, 252.0), 0.60, 1),
        (cx_g["D7"][::2], (74.0, 222.0, 128.0), 0.70, 1),
        (cx_g["PFL3"][::2], (251.0, 146.0, 60.0), 0.70, 1),
        (cx_g["EPG"][::2], (251.0, 113.0, 133.0), 0.78, 2),
    ]
    cx_layers = [
        (cx_g["ER"], (168.0, 85.0, 247.0), 0.34, 1),
        (cx_g["PEN"], (56.0, 189.0, 248.0), 0.46, 1),
        (cx_g["PEG"], (45.0, 212.0, 191.0), 0.50, 1),
        (cx_g["D7"], (74.0, 222.0, 128.0), 0.55, 1),
        (cx_g["PFL3"], (251.0, 146.0, 60.0), 0.62, 2),
        (cx_g["EPG"], (251.0, 113.0, 133.0), 0.72, 2),
    ]

    wb_scale = 0.66
    cx_scale = 2.35
    left_img, R = splat_panel(pw, ph, wb_layers, wb_center, scale=wb_scale, azim=azim, elev=elev, y_frac=0.50)
    right_img, _ = splat_panel(pw, ph, cx_layers, cx_center, scale=cx_scale, azim=azim, elev=elev, y_frac=0.52)

    full = np.full((height, width, 3), (13, 17, 23), dtype=np.uint8)
    full[52:52 + ph, :pw] = left_img
    full[52:52 + ph, pw:] = right_img

    pil = Image.fromarray(full)
    draw = ImageDraw.Draw(pil)
    f_title = ImageFont.truetype(FONT_BOLD, 14)
    f_sub = ImageFont.truetype(FONT_REG, 11)
    f_leg_b = ImageFont.truetype(FONT_BOLD, 11)
    f_leg = ImageFont.truetype(FONT_REG, 11)
    f_tag = ImageFont.truetype(FONT_MONO, 10)

    # Top and bottom bars and center divider
    draw.rectangle([(0, 0), (width, 51)], fill=(18, 24, 34))
    draw.line([(0, 51), (width, 51)], fill=(45, 55, 72), width=1)
    draw.line([(pw, 0), (pw, height)], fill=(45, 55, 72), width=1)
    draw.rectangle([(0, height - 52), (width, height)], fill=(18, 24, 34))
    draw.line([(0, height - 52), (width, height - 52)], fill=(45, 55, 72), width=1)
    draw.line([(pw, height - 52), (pw, height)], fill=(45, 55, 72), width=1)

    # Panel titles
    draw.text(
        (16, 9),
        "1. Whole-Brain Context (Optic Lobe + Central Complex)",
        fill=(241, 245, 249),
        font=f_title,
    )
    draw.text(
        (16, 30),
        "FlyWire Release 783 (139,273 neurons)  |  flyvis Optic Lobe (65 cell types) -> Central Complex",
        fill=(148, 163, 184),
        font=f_sub,
    )

    draw.text(
        (pw + 16, 9),
        "2. Central Complex Connectome (N = 457 neurons, 17,041 synapses)",
        fill=(241, 245, 249),
        font=f_title,
    )
    draw.text(
        (pw + 16, 30),
        "Reconstructed 3D arbors masked at >= 5 synapses per pair  |  4x midline zoom",
        fill=(148, 163, 184),
        font=f_sub,
    )

    # 3D anatomical landmarks in the right panel (PB, EB, BU, LAL)
    landmarks = [
        (np.array([532.7, 148.0, 182.0]), "PB", (74, 222, 128)),
        (np.array([521.7, 183.5, 82.0]), "EB", (251, 113, 133)),
        (np.array([452.0, 192.0, 78.0]), "BU", (192, 132, 252)),
        (np.array([468.0, 212.0, 68.0]), "LAL", (251, 146, 60)),
    ]
    for pt, name, col in landmarks:
        u, v, _ = project_point(pt, cx_center, R, pw, ph, cx_scale, y_frac=0.52)
        u_abs, v_abs = pw + u, 52 + v
        if pw + 25 < u_abs < width - 55 and 70 < v_abs < height - 70:
            draw_badge(draw, u_abs, v_abs, name, col, f_leg_b)

    # Bounding box indicator around CX on the left panel
    cx_u, cx_v, _ = project_point(cx_center, wb_center, R, pw, ph, wb_scale, y_frac=0.50)
    cx_v_abs = 52 + cx_v
    box_w, box_h = 54, 42
    draw.rectangle(
        [(cx_u - box_w, cx_v_abs - box_h), (cx_u + box_w, cx_v_abs + box_h)],
        outline=(251, 113, 133),
        width=1,
    )
    draw.text((cx_u - box_w, cx_v_abs - box_h - 14), "Central Complex", fill=(251, 113, 133), font=f_tag)

    # Scale bars (bottom-left of each viewport)
    wb_bar_px = int(round(100.0 * wb_scale))
    y_bar = height - 68
    draw.line([(20, y_bar), (20 + wb_bar_px, y_bar)], fill=(203, 213, 225), width=2)
    draw.text((20, y_bar - 15), "100 um", fill=(203, 213, 225), font=f_tag)

    cx_bar_px = int(round(50.0 * cx_scale))
    draw.line([(pw + 20, y_bar), (pw + 20 + cx_bar_px, y_bar)], fill=(203, 213, 225), width=2)
    draw.text((pw + 20, y_bar - 15), "50 um", fill=(203, 213, 225), font=f_tag)

    # 3D orientation triad in bottom-right of right panel
    draw_triad(draw, R, width - 64, height - 96, 22, f_tag)

    # Bottom legend bar: Left = Optic Lobe layers, Right = Central Complex cell classes
    leg_left = [
        ((110, 128, 158), "Brain shell"),
        ((56, 189, 248), "R1-8, L1-5, Mi, Tm (57 types)"),
        ((251, 191, 36), "T4a-d / T5a-d readout (8 types)"),
    ]
    lx = 16
    ly = height - 43
    draw.text((lx, ly), "Optic Lobe Pathway (flyvis, 45,669 nodes)", fill=(226, 232, 240), font=f_leg_b)
    ly2 = height - 23
    for col, text in leg_left:
        draw.rectangle([(lx, ly2 + 2), (lx + 9, ly2 + 11)], fill=col)
        draw.text((lx + 13, ly2), text, fill=(203, 213, 225), font=f_leg)
        lx += int(round(draw.textlength(text, font=f_leg))) + 24

    leg_right = [
        ((168, 85, 247), "ER (278)"),
        ((251, 113, 133), "EPG (51)"),
        ((56, 189, 248), "PEN (42)"),
        ((45, 212, 191), "PEG (20)"),
        ((74, 222, 128), "Delta7 (42)"),
        ((251, 146, 60), "PFL3 (24)"),
    ]
    rx = pw + 16
    draw.text((rx, ly), "Central Complex Circuit (ER input -> EPG/PEN/PEG/Delta7 compass -> PFL3 actor)",
              fill=(226, 232, 240), font=f_leg_b)
    for col, text in leg_right:
        draw.rectangle([(rx, ly2 + 2), (rx + 9, ly2 + 11)], fill=col)
        draw.text((rx + 13, ly2), text, fill=(203, 213, 225), font=f_leg)
        rx += int(round(draw.textlength(text, font=f_leg))) + 24

    return np.asarray(pil)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/media/fly_brain_3d.gif"))
    ap.add_argument("--frames", type=int, default=72, help="frames per full 360-degree rotation")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", default=None, help="optional single-frame PNG path")
    args = ap.parse_args(argv)

    cfg = load_config()
    base = data_dir(cfg) / "flywire"
    skel_path = base / "skeletons" / "sk_lod1_783_healed_ds2.parquet"
    ann_path = base / "raw" / cfg.connectome.flywire.annotations_file
    cx_path = base / "processed" / "cx_graph.npz"

    print("loading FlyWire 3D skeletons and annotations...")
    clouds = load_clouds(skel_path, ann_path, cx_path, seed=args.seed)

    if args.preview:
        frame = render_frame(clouds, azim=0.38, elev=0.22)
        Image.fromarray(frame).save(args.preview)
        print(f"wrote preview {args.preview}")
        return 0

    frames = []
    for i in range(args.frames):
        phase = 2.0 * np.pi * i / args.frames
        azim = phase
        elev = 0.20 * np.sin(phase) + 0.08
        frames.append(render_frame(clouds, azim=azim, elev=elev))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mp4_path = out.with_suffix(".mp4")
    imageio.mimwrite(mp4_path, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=8)

    # Quantize to a shared adaptive 256-color palette per frame with dithering off for a clean, compact GIF
    gif_frames = [
        cv2.resize(f, (960, 480), interpolation=cv2.INTER_AREA) for f in frames
    ]
    imageio.mimwrite(out, gif_frames, duration=int(round(1000 / args.fps)), loop=0)
    print(f"wrote {out} and {mp4_path} ({len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
