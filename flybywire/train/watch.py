"""Live view of PPO training: env 0's eye, the corridor from above, the EPG ring, the curves,
and optionally a 3D pybullet window that mirrors env 0's drone.

Each view runs in its own process and gets small messages through a queue. Training never waits
for them: when a queue is full the message is dropped, so a slow or closed window costs nothing.
The 3D view is a copy, not the training env itself: it rebuilds env 0's corridor and puts the
drone where env 0 reports it, without physics. (Opening a training env in a GUI tied the env to
the window: closing the window killed the env and the run, D43.)
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np

from flybywire.eval.ring import ring_activity


class Watcher:
    def __init__(self, layout, epg_index, cfg, panel: bool = True, mirror: bool = False):
        from omegaconf import OmegaConf

        ctx = mp.get_context("spawn")
        self.layout = layout
        self.epg = np.asarray(epg_index)
        c = cfg.env.corridor
        self.views, self.procs = [], []
        if panel:
            self.views.append(("panel", ctx.Queue(maxsize=32)))
            target, args = run_panel, (layout.n_wedges, layout.label, float(c.length_m), float(c.width_m))
            self._start(ctx, target, self.views[-1][1], args)
        if mirror:
            self.views.append(("mirror", ctx.Queue(maxsize=32)))
            self._start(ctx, run_mirror, self.views[-1][1], (OmegaConf.to_container(cfg, resolve=True),))

    def _start(self, ctx, target, q, args) -> None:
        proc = ctx.Process(target=target, args=(q, *args), daemon=True)
        proc.start()
        self.procs.append(proc)

    def _put(self, msg, to=("panel", "mirror")) -> None:
        for name, q in self.views:
            if name in to:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass

    def step(self, frame, r, heading, rel_yaw) -> None:
        ring = ring_activity(r.detach().cpu().numpy()[self.epg], self.layout)
        h = heading.detach().cpu().numpy()
        self._put(("step", np.asarray(frame), ring, float(np.arctan2(h[0], h[1])), float(rel_yaw)), to=("panel",))

    def position(self, pos, quat) -> None:
        self._put(("pos", np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)))

    def episode(self, obstacles) -> None:
        self._put(("episode", np.asarray(obstacles, dtype=float)))

    def update(self, stats: dict) -> None:
        keys = ("update", "success_100", "return_100", "kl_bc", "aux_heading", "value_loss", "warmup")
        self._put(("update", {k: stats.get(k) for k in keys}), to=("panel",))

    def close(self) -> None:
        self._put(None)
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def run_mirror(q, cfg_dict: dict) -> None:
    import pybullet as p
    from omegaconf import OmegaConf

    from flybywire.envs.corridor import CorridorEnv

    env = CorridorEnv(OmegaConf.create(cfg_dict), gui=True)
    cid = env._cid
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
    while p.isConnected(cid):
        msg = q.get()
        if msg is None:
            break
        if msg[0] == "episode":
            env.obstacles = msg[1]
            env._aviary.reset()
            env._drone = int(env._aviary.DRONE_IDS[0])
        elif msg[0] == "pos":
            pos, quat = msg[1], msg[2]
            p.resetBasePositionAndOrientation(env._drone, pos, quat, physicsClientId=cid)
            yaw = np.degrees(p.getEulerFromQuaternion(quat)[2])
            p.resetDebugVisualizerCamera(1.6, yaw - 90, -25, pos, physicsClientId=cid)


def run_panel(q, n_wedges: int, layout_label: str, length: float, width: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    plt.ion()
    fig = plt.figure("Fly-By-Wire PPO", figsize=(13, 7))
    grid = fig.add_gridspec(2, 3)
    ax_eye = fig.add_subplot(grid[0, 0])
    ax_map = fig.add_subplot(grid[0, 1:])
    ax_ring = fig.add_subplot(grid[1, 0], projection="polar")
    ax_succ = fig.add_subplot(grid[1, 1])
    ax_loss = fig.add_subplot(grid[1, 2])

    eye = ax_eye.imshow(np.zeros((160, 160)), cmap="gray", vmin=0, vmax=255)
    ax_eye.set_title("env 0: what the eye sees")
    ax_eye.axis("off")

    theta = (np.arange(n_wedges) + 0.5) * 2 * np.pi / n_wedges
    bars = ax_ring.bar(theta, np.zeros(n_wedges), width=2 * np.pi / n_wedges * 0.9, bottom=0.2)
    true_line, = ax_ring.plot([0, 0], [0, 1.2], "r-", lw=2, label="true heading")
    dec_line, = ax_ring.plot([0, 0], [0, 1.2], "c--", lw=2, label="decoded")
    ax_ring.set_yticks([])
    ax_ring.set_title(f"EPG ring ({layout_label})", fontsize=8)
    ax_ring.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.3, -0.1))

    trail_x, trail_y = [], []
    obstacle_patches = []
    trail, = ax_map.plot([], [], "b-", lw=1)
    drone, = ax_map.plot([], [], "bo", ms=5)
    ax_map.set_xlim(0, length)
    ax_map.set_ylim(-width / 2, width / 2)
    ax_map.set_aspect("equal")
    ax_map.axhline(width / 2, color="k")
    ax_map.axhline(-width / 2, color="k")
    ax_map.set_title("corridor from above (env 0)")

    stats = []
    last_draw = 0.0
    while plt.fignum_exists(fig.number):
        try:
            while True:
                msg = q.get_nowait()
                if msg is None:
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
                elif kind == "pos":
                    trail_x.append(msg[1][0])
                    trail_y.append(msg[1][1])
                elif kind == "episode":
                    trail_x.clear()
                    trail_y.clear()
                    for patch in obstacle_patches:
                        patch.remove()
                    obstacle_patches.clear()
                    for k, x, y, hx, hy in msg[1]:
                        patch = Circle((x, y), hx) if k == 0 else Rectangle((x - hx, y - hy), 2 * hx, 2 * hy)
                        patch.set_color("0.4")
                        obstacle_patches.append(ax_map.add_patch(patch))
                elif kind == "update":
                    stats.append(msg[1])
        except queue.Empty:
            pass
        now = time.time()
        if now - last_draw > 0.1:
            trail.set_data(trail_x, trail_y)
            if trail_x:
                drone.set_data([trail_x[-1]], [trail_y[-1]])
            if stats:
                u = [s["update"] for s in stats]
                ax_succ.clear()
                ax_succ.plot(u, [s["success_100"] for s in stats], "g-")
                ax_succ.set_ylim(0, 1)
                ax_succ.set_title("success, last 100 episodes")
                warm = [s["update"] for s in stats if s["warmup"]]
                if warm:
                    ax_succ.axvspan(min(warm), max(warm), color="0.9")
                ax_loss.clear()
                ax_loss.plot(u, [s["kl_bc"] or 0 for s in stats], label="KL to BC")
                ax_loss.plot(u, [s["aux_heading"] or 0 for s in stats], label="heading loss")
                ax_loss.legend(fontsize=7)
                ax_loss.set_xlabel("update")
            fig.canvas.draw_idle()
            last_draw = now
        plt.pause(0.03)
