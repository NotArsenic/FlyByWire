"""One policy flight stepped a decision at a time, for the lesion video and the dashboard.

Same pipeline as closed-loop evaluation (BoxEye, flyvis, core, actor mean) on a single env, plus
what the views need: the EPG ring on the anatomical layout, the decoded and true heading, the
trail, and a chase-camera image of the drone.
"""

from __future__ import annotations

import numpy as np
import pybullet as p
import torch
from omegaconf import OmegaConf

from flybywire.eval.closed_loop import load_checkpoint
from flybywire.eval.ring import bump, model_layout, ring_activity


class Flight:
    def __init__(self, ckpt, seed: int = 10000, device: str | None = None, overrides=()):
        from flybywire.clocks import clocks_from_cfg
        from flybywire.envs.corridor import CorridorEnv
        from flybywire.vision.boxeye import BoxEye
        from flybywire.vision.flyvis_stepper import FlyvisStepper

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, cfg = load_checkpoint(ckpt, self.device)
        self.cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
        self.clocks = clocks_from_cfg(self.cfg)
        self.env = CorridorEnv(self.cfg)
        self.eye = BoxEye(int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size), device=self.device)
        self.stepper = FlyvisStepper(
            model=cfg.vision.flyvis_model,
            device=self.device,
            dt=self.clocks.dt_sensory,
            cell_types=tuple(cfg.vision.cell_types),
            steady_state_seconds=float(cfg.vision.steady_state_seconds),
        )
        self.layout = model_layout(self.cfg, self.model, "anatomical")
        self.epg = torch.as_tensor(self.model.groups[self.model.group_roles["heading"]], device=self.device)
        self.lesions: tuple[str, ...] = ()
        # the EB ring has its own zero and handedness: bump angle ~ sign * heading + offset
        self.ring_sign, self.ring_offset = 1, 0.0
        self.reset(seed)

    def reset(self, seed: int) -> None:
        self.seed = seed
        self.frames, info = self.env.reset(seed=seed)
        self.obstacles = info["obstacles"]
        self.v = self.model.init_state(1)
        self.stepper.reset(1)
        self.k = 0
        self.done, self.reason = False, None
        self.trail = [info["pos"][-1][:2]]
        self.true_heading = float(info["rel_yaw"][-1])
        self.yaw = float(info["yaw"][-1])
        self.ring = np.zeros(self.layout.n_wedges)
        self.decoded = 0.0

    def align_ring(self, report) -> bool:
        """Ring frame from scripts/eval_lesions.py's intact flights, if that report exists."""
        import json
        from pathlib import Path

        if not Path(report).exists():
            return False
        ring = json.loads(Path(report).read_text())["none"]["ring"]
        self.ring_sign, self.ring_offset = ring["sign"], np.radians(ring["offset_deg"])
        return True

    def set_lesions(self, names) -> None:
        self.lesions = tuple(names)
        if self.lesions:
            self.model.lesion(self.lesions)
        else:
            self.model.restore()

    @property
    def time(self) -> float:
        return self.k * self.clocks.dt_control

    @property
    def share(self) -> float:
        return float(bump(self.ring)["fraction"])

    @torch.no_grad()
    def step(self) -> None:
        x = torch.from_numpy(self.frames).to(self.device)
        hexals = self.eye(x).half().float()
        for t in range(hexals.shape[0]):
            self.v, r = self.model.core_step(self.v, self.stepper.step(hexals[t : t + 1]))
        self.ring = ring_activity(r[0, self.epg].cpu().numpy(), self.layout)
        h = self.model.heading(r)[0].cpu().numpy()
        self.decoded = float(np.arctan2(h[0], h[1]))
        action = self.model.actor(r)[0].cpu().numpy()
        self.frames, _, term, trunc, info = self.env.step(action)
        self.k += 1
        self.trail.append(info["pos"][-1][:2])
        self.true_heading = float(info["rel_yaw"][-1])
        self.yaw = float(info["yaw"][-1])
        if term or trunc:
            self.done = True
            self.reason = ("collision" if info["collision"] else "end" if info["reached_end"]
                           else "out_of_bounds" if info["out_of_bounds"] else "time")

    @property
    def eye_frame(self) -> np.ndarray:
        return self.frames[-1]

    def chase_image(self, width: int = 400, height: int = 300) -> np.ndarray:
        """Third-person view from behind and above the drone (TinyRenderer, RGB)."""
        pos, _ = p.getBasePositionAndOrientation(self.env._drone, physicsClientId=self.env._cid)
        view = p.computeViewMatrixFromYawPitchRoll(pos, 0.8, np.degrees(self.yaw) - 90, -20, 0, 2,
                                                   physicsClientId=self.env._cid)
        proj = p.computeProjectionMatrixFOV(60, width / height, 0.05, 40, physicsClientId=self.env._cid)
        img = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER, shadow=0,
                               physicsClientId=self.env._cid)[2]
        return np.reshape(img, (height, width, 4))[..., :3]

    def close(self) -> None:
        self.model.restore()
        self.env.close()


def draw(fig, flight: Flight, title: str = "") -> None:
    """Chase view, eye, corridor map and EPG ring of the current state on a cleared figure."""
    from matplotlib.patches import Circle, Rectangle

    fig.clear()
    grid = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], left=0.02, right=0.98, top=0.93, bottom=0.03,
                            wspace=0.08, hspace=0.18)
    ax_chase = fig.add_subplot(grid[0, :2])
    ax_ring = fig.add_subplot(grid[0, 2], projection="polar")
    ax_eye = fig.add_subplot(grid[1, 0])
    ax_map = fig.add_subplot(grid[1, 1:])

    ax_chase.imshow(flight.chase_image())
    ax_chase.axis("off")
    lesion = "+".join(flight.lesions) if flight.lesions else "none"
    ax_chase.set_title(f"{title}t = {flight.time:4.1f} s    lesion: {lesion}", loc="left", fontsize=13,
                       color="crimson" if flight.lesions else "black")

    ax_eye.imshow(flight.eye_frame, cmap="gray", vmin=0, vmax=255)
    ax_eye.set_title("what the eye sees", fontsize=9)
    ax_eye.axis("off")

    c = flight.cfg.env.corridor
    for kind, x, y, hx, hy in flight.obstacles:
        ax_map.add_patch(Circle((x, y), hx, color="0.45") if kind == 0
                         else Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, color="0.45"))
    trail = np.asarray(flight.trail)
    ax_map.plot(trail[:, 0], trail[:, 1], "b-", lw=1)
    ax_map.plot(*trail[-1], "bo", ms=4)
    ax_map.arrow(*trail[-1], 0.8 * np.cos(flight.yaw), 0.8 * np.sin(flight.yaw), color="b", width=0.03)
    ax_map.axhline(c.width_m / 2, color="k")
    ax_map.axhline(-c.width_m / 2, color="k")
    ax_map.set_xlim(0, c.length_m)
    ax_map.set_ylim(-c.width_m / 2 - 0.1, c.width_m / 2 + 0.1)
    ax_map.set_aspect("equal")
    ax_map.set_yticks([])
    ax_map.set_title("corridor from above", fontsize=9)

    n = flight.layout.n_wedges
    ring = np.nan_to_num(flight.ring)
    vmax = max(ring.max(), 1e-6)
    theta = (np.arange(n) + 0.5) * 2 * np.pi / n
    ax_ring.bar(theta, ring / vmax, width=2 * np.pi / n * 0.92, bottom=0.25, color=_viridis(ring / vmax))
    on_ring = [flight.ring_sign * a + flight.ring_offset for a in (flight.true_heading, flight.decoded)]
    ax_ring.plot([on_ring[0]] * 2, [0, 1.3], "r-", lw=2, label="true heading")
    ax_ring.plot([on_ring[1]] * 2, [0, 1.3], "c--", lw=2, label="decoded from EPG")
    ax_ring.set_ylim(0, 1.35)
    ax_ring.set_yticks([])
    ax_ring.set_xticks([])
    ax_ring.set_title(f"EPG ring, anatomical EB layout\nbump share {flight.share:.2f}", fontsize=9)
    ax_ring.legend(loc="lower center", fontsize=7, bbox_to_anchor=(0.5, -0.32), ncol=2)


def _viridis(x):
    import matplotlib.pyplot as plt

    return plt.cm.viridis(np.clip(x, 0, 1))


def figure_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
