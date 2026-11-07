#!/usr/bin/env python
"""Render the lesion demo: one held-out corridor flight with cell types silenced at fixed times.

Each frame (one per decision, 25 fps, real time) shows a chase camera, what the eye sees, the
corridor from above and the EPG ring on the anatomical EB layout. The heading arrows are drawn in
the ring's frame (bump angle ~ sign * heading + offset, fitted on the intact flights of
scripts/eval_lesions.py when its report is there), so with an intact ring the bump sits under
the red arrow.

--schedule lists lesions as NAME:START-END in seconds (NAME may join groups with +); outside them
the network is intact. The flight runs until the episode ends; --seconds sets the episode length
(the corridor is 30 m, about 20 s of flight).

Usage:
  python scripts/render_lesion_video.py [--ckpt runs/ppo/demo/model_best.pt] [--out docs/media/lesion_demo.mp4]
                                        [--seed 10003] [--seconds 20] [--schedule PEN:5-9 D7:11-14 EPG:16-19]
                                        [--gif docs/media/lesion_demo.gif]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flybywire.config import REPO_ROOT
from flybywire.gif import save_gif


def parse_schedule(items) -> list[tuple[tuple[str, ...], float, float]]:
    out = []
    for item in items:
        name, span = item.split(":")
        start, end = (float(s) for s in span.split("-"))
        out.append((tuple(name.split("+")), start, end))
    return out


def active(schedule, t: float) -> tuple[str, ...]:
    for names, start, end in schedule:
        if start <= t < end:
            return names
    return ()


def main(argv=None) -> int:
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from flybywire.eval.live import Flight, draw, figure_rgb

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "runs/sweep_dart/real_anatomical_s5/model_best.pt"),
    )
    ap.add_argument("--out", default=str(REPO_ROOT / "docs/media/lesion_demo.mp4"))
    ap.add_argument(
        "--gif",
        default=None,
        help="also write a GIF (full size, every 2nd frame)",
    )
    ap.add_argument("--seed", type=int, default=10003)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument(
        "--schedule", nargs="*", default=["PEN:5-9", "D7:11-14", "EPG:16-19"]
    )
    ap.add_argument(
        "overrides",
        nargs="*",
        help="key=value on the corridor, e.g. a curriculum level",
    )
    ap.add_argument(
        "--lesion-report",
        default=None,
        help="lesions.json for the ring frame (default runs/lesions/<run>/lesions.json)",
    )
    args = ap.parse_args(argv)

    schedule = parse_schedule(args.schedule)
    flight = Flight(
        args.ckpt,
        args.seed,
        overrides=[f"env.episode_seconds={args.seconds}", *args.overrides],
    )
    report = Path(
        args.lesion_report
        or REPO_ROOT / "runs/lesions" / Path(args.ckpt).parent.name / "lesions.json"
    )
    if not flight.align_ring(report):
        print(f"no {report}: heading arrows are not aligned to the ring")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 7.2), dpi=100)
    frames = []
    while not flight.done:
        names = active(schedule, flight.time)
        if names != flight.lesions:
            flight.set_lesions(names)
        flight.step()
        draw(fig, flight)
        frames.append(figure_rgb(fig))
    plt.close(fig)
    end = f"{flight.reason} at {flight.time:.1f} s"
    flight.close()

    fps = round(1 / flight.clocks.dt_control)
    imageio.mimwrite(
        out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8
    )
    print(f"wrote {out}: {len(frames)} frames, episode ended by {end}")
    if args.gif:
        save_gif(frames[::2], args.gif, fps / 2)  # half the frame rate keeps the file a few MB
        print(f"wrote {args.gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
