"""Loading / validating the calibrated T4/T5 preferred-direction map.

``data/flyvis/direction_map.json`` is written by ``scripts/calibrate_t4t5_directions.py``.
Directions are unit vectors in image coordinates (x right, y down), measured with drifting
gratings presented through the same BoxEye used everywhere else.

Schema::

    {
      "model": "flow/0000/000", "dt": 0.01, "convention": "image x right, y down",
      "subtypes": {"T4a": {"direction": [dx, dy], "angle_deg": ..., "dsi": ..., "responses": {...}}, ...},
      "baseline": {"T4a": <mean relu(activity) under grey>, ...},
      ...
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from flybywire.vision.flyvis_stepper import DEFAULT_CELL_TYPES


def default_direction_map_path(cfg=None) -> Path:
    from flybywire.paths import setup_flyvis_env

    return setup_flyvis_env() / "direction_map.json"


def load_direction_map(path: str | Path | None = None, cell_types=DEFAULT_CELL_TYPES) -> dict:
    path = Path(path) if path else default_direction_map_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make calibrate` (scripts/calibrate_t4t5_directions.py)")
    data = json.loads(path.read_text())
    missing = [t for t in cell_types if t not in data.get("subtypes", {})]
    if missing:
        raise ValueError(f"{path} lacks subtypes {missing}")
    return data


def direction_vectors(data: dict, cell_types=DEFAULT_CELL_TYPES) -> np.ndarray:
    """(8, 2) unit preferred-direction vectors in image coordinates."""
    e = np.array([data["subtypes"][t]["direction"] for t in cell_types], dtype=np.float64)
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("degenerate (zero) preferred direction in direction map")
    return e / norms


def baselines(data: dict, cell_types=DEFAULT_CELL_TYPES) -> np.ndarray:
    """(8,) grey-steady-state baseline of relu(activity) per subtype (0 if absent)."""
    b = data.get("baseline", {})
    return np.array([float(b.get(t, 0.0)) for t in cell_types], dtype=np.float64)


def measure_directions(
    stepper,
    eye,
    cell_types=DEFAULT_CELL_TYPES,
    frame_size: int = 160,
    frame_hz: float = 100.0,
    n_directions: int = 12,
    conditions=((40.0, 2.0), (40.0, 4.0), (64.0, 2.0), (64.0, 4.0)),
    seconds: float = 1.5,
    discard_seconds: float = 0.5,
    central_radius: float = 0.6,
) -> dict:
    """Measure each T4/T5 subtype's preferred image direction with drifting square-wave gratings.

    Gratings drift in ``n_directions`` directions evenly spaced from 0° (+x); with the default 12
    (every 30°) this includes the four cardinal directions +x (0°), +y (90°, down), -x (180°) and
    -y (270°). For each (period_px, temporal_hz) condition all directions run as one batch from the
    grey steady state; the rectified activity ``relu(v)`` is averaged over central hexals
    (normalised radius < ``central_radius``) and over time after ``discard_seconds``, then averaged
    over conditions. Preferred direction = angle of the vector sum ``Σ_θ R(θ) e^{iθ}``;
    ``dsi = |Σ_θ R(θ) e^{iθ}| / Σ_θ R(θ)``. (docs/DECISIONS.md D23.)
    """
    import torch

    from flybywire.vision.flowfields import hexal_normalized_coords
    from flybywire.vision.stimuli import grating_sequence

    n_frames = int(round(seconds * frame_hz))
    n_skip = int(round(discard_seconds * frame_hz))
    angles = np.arange(n_directions) * 2 * np.pi / n_directions
    xy = hexal_normalized_coords(eye.extent, eye.kernel_size)
    central = torch.as_tensor(np.linalg.norm(xy, axis=1) < central_radius, device=stepper.device)
    R = np.zeros((n_directions, len(cell_types)))
    for period, tf in conditions:
        frames = np.stack([grating_sequence(frame_size, period, tf, n_frames, frame_hz, float(a)) for a in angles])
        x = torch.from_numpy(frames).to(eye.device)  # (D, T, H, W)
        hexals = torch.stack([eye(x[:, t]) for t in range(n_frames)], dim=1)  # (D, T, 721)
        out = stepper.run_sequence(hexals)  # (D, T, 8, 721)
        R += torch.relu(out[:, n_skip:][..., central]).mean(dim=(1, 3)).cpu().numpy() / len(conditions)
    stepper.reset(1)
    grey = torch.relu(stepper.state[:, stepper._out_index][..., central]).mean(dim=(0, 2)).cpu().numpy()
    vec = (R * np.exp(1j * angles)[:, None]).sum(axis=0)  # (8,) complex
    cardinal = {"+x": 0.0, "+y": 90.0, "-x": 180.0, "-y": 270.0}
    deg = np.round(np.degrees(angles), 6)
    subtypes = {}
    for i, t in enumerate(cell_types):
        norm = float(np.abs(vec[i]))
        d = vec[i] / norm if norm > 0 else 0.0
        subtypes[t] = {
            "direction": [float(np.real(d)), float(np.imag(d))],
            "angle_deg": float(np.degrees(np.angle(vec[i]))),
            "dsi": float(norm / R[:, i].sum()) if R[:, i].sum() > 0 else 0.0,
            "responses": {k: float(R[np.flatnonzero(deg == a)[0], i]) for k, a in cardinal.items() if a in deg},
            "tuning_curve": {f"{a:g}": float(r) for a, r in zip(deg, R[:, i], strict=True)},
        }
    return {
        "model": getattr(stepper, "model_id", None),
        "dt": getattr(stepper, "dt", None),
        "convention": "image coordinates: x right, y down; angle_deg = atan2(dy, dx); +y (90°) is downward",
        "stimulus": {
            "type": "square-wave grating, full contrast",
            "n_directions": n_directions,
            "conditions_period_px_temporal_hz": [list(c) for c in conditions],
            "seconds": seconds,
            "discard_seconds": discard_seconds,
            "frame_size": frame_size,
            "central_radius": central_radius,
            "response": "mean relu(activity) over central hexals and time, averaged over conditions",
            "estimate": "vector sum over directions",
        },
        "subtypes": subtypes,
        "baseline": {t: float(grey[i]) for i, t in enumerate(cell_types)},
    }
