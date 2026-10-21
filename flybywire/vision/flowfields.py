"""Ego-motion optic-flow templates (Longuet-Higgins & Prazdny 1980) at hexal / pixel positions."""

from __future__ import annotations

import numpy as np

CHANNELS = (
    "yaw",
    "pitch",
    "roll",
    "forward",
    "lateral",
    "vertical",
    "loom_left",
    "loom_right",
)


def hex_coords(extent: int) -> tuple[np.ndarray, np.ndarray]:
    """flyvis ``get_hex_coords`` (u, v), same order as BoxEye output and T4/T5 cells."""
    from flybywire.paths import import_flyvis

    try:
        import_flyvis()
        from flyvis.utils.hex_utils import get_hex_coords
    except ImportError:  # pragma: no cover - identical fallback
        u, v = [], []
        for q in range(-extent, extent + 1):
            for r in range(max(-extent, -extent - q), min(extent, extent - q) + 1):
                u.append(q)
                v.append(r)
        return np.array(u), np.array(v)
    u, v = get_hex_coords(extent)
    return np.asarray(u), np.asarray(v)


def hexal_normalized_coords(extent: int = 15, kernel_size: int = 13) -> np.ndarray:
    """(n_hexals, 2) normalised image coords in [-1, 1]: column 0 = x (right), 1 = y (down)."""
    u, v = hex_coords(extent)
    row = kernel_size * (u + v / 2.0)
    col = kernel_size * v.astype(float)
    h = int(row.max() - row.min()) + 1
    w = int(col.max() - col.min()) + 1
    y = ((row + h // 2) + 0.5) / h * 2 - 1
    x = ((col + w // 2) + 0.5) / w * 2 - 1
    return np.stack([x, y], axis=1)


def normalized_to_focal(xy_norm: np.ndarray, fov_deg: float) -> np.ndarray:
    return xy_norm * np.tan(np.deg2rad(fov_deg) / 2.0)


def body_to_camera_rotation(tilt_deg: float = 0.0) -> np.ndarray:
    """R such that v_cam = R @ v_body. Rows are camera axes (X right, Y down, Z fwd) in body FLU."""
    t = np.deg2rad(tilt_deg)
    return np.array(
        [
            [0.0, -1.0, 0.0],  # X_cam (right) = -y_body
            [-np.sin(t), 0.0, -np.cos(t)],  # Y_cam (down)
            [np.cos(t), 0.0, -np.sin(t)],  # Z_cam (forward, pitched down by tilt)
        ]
    )


def egomotion_flow(
    xy: np.ndarray, T_cam: np.ndarray, W_cam: np.ndarray, depth: float = 1.0
) -> np.ndarray:
    """Image flow (n, 2) at focal-unit points ``xy`` for camera translation T and rotation W."""
    x, y = xy[:, 0], xy[:, 1]
    tx, ty, tz = T_cam
    wx, wy, wz = W_cam
    u = (-tx + x * tz) / depth + wx * x * y - wy * (1 + x**2) + wz * y
    v = (-ty + y * tz) / depth + wx * (1 + y**2) - wy * x * y - wz * x
    return np.stack([u, v], axis=1)


def channel_flow(xy: np.ndarray, channel: str, tilt_deg: float = 0.0) -> np.ndarray:
    """Flow field for unit positive self-motion in ``channel`` (body frame), at focal-unit points."""
    R = body_to_camera_rotation(tilt_deg)
    zero = np.zeros(3)
    axes = {
        "x": np.array([1.0, 0, 0]),
        "y": np.array([0, 1.0, 0]),
        "z": np.array([0, 0, 1.0]),
    }
    if channel in ("yaw", "pitch", "roll"):
        ax = {"roll": "x", "pitch": "y", "yaw": "z"}[channel]
        return egomotion_flow(xy, zero, R @ axes[ax])
    if channel in ("forward", "lateral", "vertical"):
        ax = {"forward": "x", "lateral": "y", "vertical": "z"}[channel]
        return egomotion_flow(xy, R @ axes[ax], zero)
    if channel in ("loom_left", "loom_right"):
        f = egomotion_flow(xy, R @ axes["x"], zero)
        keep = xy[:, 0] < 0 if channel == "loom_left" else xy[:, 0] > 0
        return f * keep[:, None]
    raise ValueError(f"unknown channel {channel!r}")


def flow_templates(
    xy: np.ndarray, channels=CHANNELS, tilt_deg: float = 0.0
) -> np.ndarray:
    """(n_channels, n_points, 2) flow fields."""
    return np.stack([channel_flow(xy, c, tilt_deg) for c in channels])


def hexal_flow_templates(
    extent: int = 15,
    kernel_size: int = 13,
    fov_deg: float = 110.0,
    tilt_deg: float = 0.0,
    channels=CHANNELS,
) -> np.ndarray:
    xy = normalized_to_focal(hexal_normalized_coords(extent, kernel_size), fov_deg)
    return flow_templates(xy, channels, tilt_deg)


def pixel_grid_focal(size: int, fov_deg: float) -> np.ndarray:
    """(size*size, 2) focal-unit coords of pixel centres of a square frame, row-major."""
    c = (np.arange(size) + 0.5) / size * 2 - 1
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return normalized_to_focal(np.stack([xx.ravel(), yy.ravel()], axis=1), fov_deg)
