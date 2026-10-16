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
    periods_px=(24.0, 40.0),
    temporal_hz: float = 2.0,
    seconds: float = 1.5,
    discard_seconds: float = 0.5,
) -> dict:
    """Measure each T4/T5 subtype's preferred image direction with drifting square-wave gratings.

    For every direction in {+x, -x, +y, -y} and every grating period, the stepper is reset to the
    grey steady state, the grating is presented for ``seconds`` and the rectified activity
    ``relu(v)`` is averaged over hexals and over time after ``discard_seconds``. The preferred
    vector is ``(R(+x) - R(-x), R(+y) - R(-y))`` (averaged over periods), normalised.
    ``dsi = |vector| / (sum of the four responses)``.
    """
    import torch

    from flybywire.vision.stimuli import grating_sequence

    n_frames = int(round(seconds * frame_hz))
    n_skip = int(round(discard_seconds * frame_hz))
    resp = {d: np.zeros(len(cell_types)) for d in ("+x", "-x", "+y", "-y")}
    for d in resp:
        for period in periods_px:
            frames = grating_sequence(frame_size, period, temporal_hz, n_frames, frame_hz, d)
            hexals = eye(torch.from_numpy(frames).to(eye.device))  # (T, 721)
            out = stepper.run_sequence(hexals[None])  # (1, T, 8, 721)
            r = torch.relu(out[0, n_skip:]).mean(dim=(0, 2)).cpu().numpy()  # (8,)
            resp[d] += r / len(periods_px)
    stepper.reset(1)
    grey = torch.relu(stepper.state[:, stepper._out_index]).mean(dim=(0, 2)).cpu().numpy()
    subtypes = {}
    for i, t in enumerate(cell_types):
        vec = np.array([resp["+x"][i] - resp["-x"][i], resp["+y"][i] - resp["-y"][i]])
        total = sum(resp[d][i] for d in resp)
        norm = float(np.linalg.norm(vec))
        subtypes[t] = {
            "direction": (vec / norm).tolist() if norm > 0 else [0.0, 0.0],
            "angle_deg": float(np.degrees(np.arctan2(vec[1], vec[0]))),
            "dsi": float(norm / total) if total > 0 else 0.0,
            "responses": {d: float(resp[d][i]) for d in resp},
        }
    return {
        "model": getattr(stepper, "model_id", None),
        "dt": getattr(stepper, "dt", None),
        "convention": "image coordinates: x right, y down; angle_deg = atan2(dy, dx)",
        "stimulus": {
            "type": "square-wave grating",
            "periods_px": list(periods_px),
            "temporal_hz": temporal_hz,
            "seconds": seconds,
            "discard_seconds": discard_seconds,
            "frame_size": frame_size,
            "response": "mean relu(activity) over hexals and time",
        },
        "subtypes": subtypes,
        "baseline": {t: float(grey[i]) for i, t in enumerate(cell_types)},
    }
