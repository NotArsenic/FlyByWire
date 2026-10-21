#!/usr/bin/env python
"""Gate 1: can frozen flyvis be stepped frame by frame, with state carried, fast enough?

Equivalence: 200 frames of drifting texture (160x160 -> BoxEye hexals) go through the frozen
stepper two ways, (a) one ``Network.forward`` call over the whole sequence and (b) frame by frame
with carried state, in both stepper modes:

  * ``public``: ``Network.forward(x_t[:, None], dt, state=prev, as_states=True)[-1]``
  * ``fast``: parameters computed once, ``Network._next_state(params, state, x_t, dt)``

A max |diff| / max |response| above 1e-5 is a failure. Throughput is timed at B = 1 and B = 12
(50 warm-up + 500 timed steps, ``torch.cuda.synchronize()`` around the timing).

Passes if state carries correctly and B = 12 reaches >= 100 batched steps/s in the default
(``fast``) mode. Writes ``runs/gate1/report.json``. If it fails the script stops and reports; it
never switches to the Farneback backend by itself.
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.config import load_config, runs_dir
from flybywire.vision.boxeye import BoxEye
from flybywire.vision.flyvis_stepper import FlyvisStepper
from flybywire.vision.stimuli import panorama_texture, view_from_panorama

REL_TOL = 1e-5
MIN_STEPS_PER_S_B12 = 100.0


def drifting_frames(n: int, size: int, fov: float, hz: float, yaw_rate: float = 1.0, seed: int = 0) -> np.ndarray:
    pano = panorama_texture(size, fov, seed=seed)
    return np.stack([view_from_panorama(pano, yaw_rate * t / hz, size, fov) for t in range(n)])


def throughput(stepper: FlyvisStepper, hexal_frame: torch.Tensor, batch: int, warmup=50, timed=500) -> dict:
    h = hexal_frame.expand(batch, -1).contiguous()
    stepper.reset(batch)
    for _ in range(warmup):
        stepper.step(h)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        stepper.step(h)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return {"batch": batch, "steps": timed, "seconds": dt, "batched_steps_per_s": timed / dt,
            "env_steps_per_s": timed * batch / dt}


def boxeye_throughput(eye: BoxEye, frame: np.ndarray, batch: int, warmup=20, timed=200) -> float:
    x = torch.from_numpy(np.repeat(frame[None], batch, 0)).to(eye.device)
    for _ in range(warmup):
        eye(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed):
        eye(x)
    torch.cuda.synchronize()
    return timed / (time.perf_counter() - t0)


def main(argv=None) -> int:
    cfg = load_config(overrides=argv if argv is not None else sys.argv[1:])
    if not torch.cuda.is_available():
        print("Gate 1 needs a CUDA GPU.")
        return 2
    clocks = clocks_from_cfg(cfg)
    size, fov = int(cfg.env.frame_size), float(cfg.env.camera_fov_deg)
    eye = BoxEye(int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size), device="cuda")
    stepper = FlyvisStepper(model=cfg.vision.flyvis_model, device="cuda", dt=clocks.dt_sensory,
                            cell_types=tuple(cfg.vision.cell_types),
                            steady_state_seconds=float(cfg.vision.steady_state_seconds), mode="fast")
    assert not stepper.requires_grad_any(), "flyvis parameters must not require grad"
    assert not stepper.net.training, "flyvis must be in eval mode"

    frames = drifting_frames(200, size, fov, clocks.sensory_hz)
    hexals = eye(torch.from_numpy(frames).cuda())[None]
    ref = stepper.full_sequence_reference(hexals)
    scale = float(ref.abs().max())
    equivalence = {}
    for mode in ("fast", "public"):
        stepper.mode = mode
        out = stepper.run_sequence(hexals)
        max_abs = float((out - ref).abs().max())
        equivalence[mode] = {"max_abs_diff": max_abs, "rel_diff": max_abs / scale, "pass": max_abs / scale <= REL_TOL}
    stepper.mode = "fast"

    perf = {}
    for mode in ("fast", "public"):
        stepper.mode = mode
        perf[mode] = [throughput(stepper, hexals[0, 0:1], b) for b in (1, 12)]
    stepper.mode = "fast"
    box = {f"B{b}_frames_per_call_per_s": boxeye_throughput(eye, frames[0], b) for b in (1, 12)}

    used = stepper.mode
    b12 = next(p for p in perf[used] if p["batch"] == 12)["batched_steps_per_s"]
    passed = bool(equivalence[used]["pass"] and b12 >= MIN_STEPS_PER_S_B12)
    report = {
        "gate": 1,
        "result": "PASS" if passed else "FAIL",
        "criteria": {"rel_tol": REL_TOL, "min_batched_steps_per_s_B12": MIN_STEPS_PER_S_B12, "stepper_mode": used},
        "model": cfg.vision.flyvis_model,
        "dt": clocks.dt_sensory,
        "n_frames": int(hexals.shape[1]),
        "response_scale_max_abs": scale,
        "equivalence": equivalence,
        "throughput": perf,
        "boxeye_calls_per_s": box,
        "env": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                "python": platform.python_version()},
    }
    out_dir = runs_dir(cfg) / "gate1"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\n=== Gate 1: flyvis state carry + throughput ===")
    print(f"response scale max|a| = {scale:.4f}")
    for mode, e in equivalence.items():
        print(f"  equivalence [{mode:6s}] max|a-b| = {e['max_abs_diff']:.3e}  rel = {e['rel_diff']:.3e}  "
              f"{'ok' if e['pass'] else 'FAIL'}")
    for mode, rows in perf.items():
        for r in rows:
            print(f"  throughput [{mode:6s}] B={r['batch']:2d}: {r['batched_steps_per_s']:8.1f} batched steps/s  "
                  f"{r['env_steps_per_s']:9.1f} env-steps/s")
    print(f"  BoxEye: {box}")
    print(f"RESULT: {report['result']}  (mode={used}, B=12 {b12:.1f} steps/s, need >= {MIN_STEPS_PER_S_B12})")
    print(f"wrote {out_dir / 'report.json'}")
    if not passed:
        print("Gate 1 FAILED. Not switching backends automatically (vision.backend=farneback is the fallback).")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
