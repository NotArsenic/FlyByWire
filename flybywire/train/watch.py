"""Live view of PPO training in one window: a chase camera on env 0's drone, what its eye sees, the corridor from above, the EPG ring and the training curves."""

from __future__ import annotations

import multiprocessing as mp
import queue
import shutil
import time
from pathlib import Path

import numpy as np

from flybywire.eval.ring import ring_activity


def room_to_record(folder, budget_gb: float, min_free_gb: float = 10.0) -> bool:
    used = sum(f.stat().st_size for f in Path(folder).glob("*.mp4"))
    return used < budget_gb * 1e9 and shutil.disk_usage(folder).free > min_free_gb * 1e9


class Watcher:
    def __init__(self, layout, epg_index, cfg, chase: bool = True):
        from omegaconf import OmegaConf

        ctx = mp.get_context("spawn")
        self.q = ctx.Queue(maxsize=64)
        self.layout = layout
        self.epg = np.asarray(epg_index)
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) if chase else None
        self.proc = ctx.Process(
            target=run_window,
            args=(
                self.q,
                layout.n_wedges,
                layout.label,
                cfg_dict,
                float(cfg.env.corridor.length_m),
                float(cfg.env.corridor.width_m),
            ),
            daemon=True,
        )
        self.proc.start()

    def _put(self, msg) -> None:
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            pass

    def step(self, frame, r, heading, rel_yaw) -> None:
        ring = ring_activity(r.detach().cpu().numpy()[self.epg], self.layout)
        h = heading.detach().cpu().numpy()
        self._put(
            (
                "step",
                np.asarray(frame),
                ring,
                float(np.arctan2(h[0], h[1])),
                float(rel_yaw),
            )
        )

    def position(self, pos, quat) -> None:
        self._put(("pos", np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)))

    def episode(self, obstacles) -> None:
        self._put(("episode", np.asarray(obstacles, dtype=float)))

    def update(self, stats: dict) -> None:
        keys = (
            "update",
            "success_100",
            "return_100",
            "kl_bc",
            "aux_heading",
            "value_loss",
            "warmup",
        )
        self._put(("update", {k: stats.get(k) for k in keys}))

    def record(self, path, title: str) -> None:

        self.q.put(("record", str(path), title))

    def close(self) -> None:
        try:
            self.q.put(None, timeout=5)
        except queue.Full:
            pass
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.terminate()


class Mirror:
    """env 0's corridor in a headless pybullet client, for the chase camera."""

    def __init__(self, cfg_dict: dict):
        from omegaconf import OmegaConf

        from flybywire.envs.corridor import CorridorEnv

        self.env = CorridorEnv(OmegaConf.create(cfg_dict))
        self.pos, self.yaw = np.array([1.0, 0.0, 1.2]), 0.0

    def episode(self, obstacles) -> None:
        self.env.obstacles = obstacles
        self.env._aviary.reset()
        self.env._drone = int(self.env._aviary.DRONE_IDS[0])

    def pose(self, pos, quat) -> None:
        import pybullet as p

        self.pos = pos
        self.yaw = p.getEulerFromQuaternion(quat)[2]
        p.resetBasePositionAndOrientation(
            self.env._drone, pos, quat, physicsClientId=self.env._cid
        )

    def image(self) -> np.ndarray:
        from flybywire.eval.live import chase_image

        return chase_image(self.env._cid, self.pos, self.yaw, 480, 300)


def run_window(
    q,
    n_wedges: int,
    layout_label: str,
    cfg_dict: dict | None,
    length: float,
    width: float,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    mirror = Mirror(cfg_dict) if cfg_dict is not None else None
    plt.ion()
    fig = plt.figure("Fly-By-Wire PPO", figsize=(16, 8.4))
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.25, 1],
        left=0.02,
        right=0.98,
        top=0.93,
        bottom=0.06,
        wspace=0.15,
        hspace=0.25,
    )
    ax_chase = fig.add_subplot(grid[0, :2])
    ax_eye = fig.add_subplot(grid[0, 2])
    ax_ring = fig.add_subplot(grid[0, 3], projection="polar")
    ax_map = fig.add_subplot(grid[1, :2])
    ax_succ = fig.add_subplot(grid[1, 2])
    ax_loss = fig.add_subplot(grid[1, 3])
    title = fig.suptitle("", fontsize=13)

    chase = ax_chase.imshow(np.zeros((300, 480, 3), np.uint8))
    ax_chase.set_title("env 0, chase camera", fontsize=9)
    ax_chase.axis("off")
    eye = ax_eye.imshow(np.zeros((160, 160)), cmap="gray", vmin=0, vmax=255)
    ax_eye.set_title("what the eye sees", fontsize=9)
    ax_eye.axis("off")

    theta = (np.arange(n_wedges) + 0.5) * 2 * np.pi / n_wedges
    bars = ax_ring.bar(
        theta, np.zeros(n_wedges), width=2 * np.pi / n_wedges * 0.9, bottom=0.2
    )
    (true_line,) = ax_ring.plot([0, 0], [0, 1.2], "r-", lw=2, label="true heading")
    (dec_line,) = ax_ring.plot([0, 0], [0, 1.2], "c--", lw=2, label="decoded")
    ax_ring.set_yticks([])
    ax_ring.set_xticks([])
    ax_ring.set_title(f"EPG ring ({layout_label})", fontsize=8)
    ax_ring.legend(loc="lower center", fontsize=7, bbox_to_anchor=(0.5, -0.25), ncol=2)

    trail_x, trail_y, obstacle_patches = [], [], []
    (trail,) = ax_map.plot([], [], "b-", lw=1)
    (drone,) = ax_map.plot([], [], "bo", ms=5)
    ax_map.set_xlim(0, length)
    ax_map.set_ylim(-width / 2, width / 2)
    ax_map.set_aspect("equal")
    ax_map.axhline(width / 2, color="k")
    ax_map.axhline(-width / 2, color="k")
    ax_map.set_yticks([])
    ax_map.set_title("corridor from above", fontsize=9)

    import imageio.v2 as imageio

    stats, last_draw, closed = [], 0.0, False
    armed, recording, writer = None, None, None

    def finish_recording():
        if writer is not None:
            writer.close()

    while True:
        new_step = False
        try:
            while True:
                msg = q.get_nowait()
                if msg is None:
                    finish_recording()
                    plt.close(fig)
                    return
                kind = msg[0]
                if kind == "step":
                    _, frame, ring, decoded, true = msg
                    eye.set_data(frame)
                    ring = np.nan_to_num(ring)
                    top = max(ring.max(), 1e-6)
                    for b, h in zip(bars, ring / top, strict=True):
                        b.set_height(h)
                        b.set_color(plt.cm.viridis(h))
                    true_line.set_xdata([true, true])
                    dec_line.set_xdata([decoded, decoded])
                    new_step = True
                    if recording is not None:
                        break  # one video frame per decision
                elif kind == "pos":
                    trail_x.append(msg[1][0])
                    trail_y.append(msg[1][1])
                    if mirror is not None:
                        mirror.pose(msg[1], msg[2])
                elif kind == "episode":
                    finish_recording()
                    recording, writer = armed, None
                    if armed is not None:
                        writer = imageio.get_writer(
                            armed[0],
                            fps=25,
                            codec="libx264",
                            quality=7,
                            macro_block_size=8,
                        )
                        armed = None
                    trail_x.clear()
                    trail_y.clear()
                    for patch in obstacle_patches:
                        patch.remove()
                    obstacle_patches.clear()
                    for k, x, y, hx, hy in msg[1]:
                        patch = (
                            Circle((x, y), hx)
                            if k == 0
                            else Rectangle((x - hx, y - hy), 2 * hx, 2 * hy)
                        )
                        patch.set_color("0.4")
                        obstacle_patches.append(ax_map.add_patch(patch))
                    if mirror is not None:
                        mirror.episode(msg[1])
                elif kind == "update":
                    stats.append(msg[1])
                elif kind == "record":
                    armed = (msg[1], msg[2])
        except queue.Empty:
            pass

        now = time.time()
        if now - last_draw > 0.1 or (recording is not None and new_step):
            if mirror is not None:
                chase.set_data(mirror.image())
            trail.set_data(trail_x, trail_y)
            if trail_x:
                drone.set_data([trail_x[-1]], [trail_y[-1]])
            if stats:
                u = [s["update"] for s in stats]
                ax_succ.clear()
                ax_succ.plot(u, [s["success_100"] for s in stats], "g-")
                ax_succ.set_ylim(0, 1)
                ax_succ.set_title("success, last 100 training episodes", fontsize=9)
                warm = [s["update"] for s in stats if s["warmup"]]
                if warm:
                    ax_succ.axvspan(min(warm), max(warm), color="0.9")
                ax_loss.clear()
                ax_loss.plot(u, [s["kl_bc"] or 0 for s in stats], label="KL to BC")
                ax_loss.plot(
                    u, [s["aux_heading"] or 0 for s in stats], label="heading loss"
                )
                ax_loss.legend(fontsize=7)
                ax_loss.set_xlabel("update")
                title.set_text(
                    f"PPO update {u[-1]}"
                    + (f"   (recording: {recording[1]})" if recording else "")
                )
            if recording is not None:
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            elif not closed:
                fig.canvas.draw_idle()
            last_draw = now
        if closed:
            time.sleep(0.02)
        elif plt.fignum_exists(fig.number):
            plt.pause(0.02)
        else:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            FigureCanvasAgg(
                fig
            )  # the window was closed: keep drawing off-screen for recordings
            closed = True
