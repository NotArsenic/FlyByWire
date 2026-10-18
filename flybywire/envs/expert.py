"""Scripted expert that generates the BC data. It reads the privileged state in info.

A potential field in the corridor plane pulls toward a point straight down the corridor, or toward
the wider gap beside the next obstacle when that obstacle blocks the centre line, and pushes away
from obstacle and wall surfaces closer than influence_m. The field direction plus a slow random
wander (seeded OU process, so dodges differ between episodes) is the desired heading, and a
heading PID turns toward it. Velocity is `speed` along that heading, altitude is held.

It acts on the last frame of each window, which is what decision k sees.
"""

from __future__ import annotations

import numpy as np

from flybywire.clocks import clocks_from_cfg
from flybywire.envs.corridor import normalize_action, wrap


def surface_distance(px: float, py: float, obstacle) -> tuple[float, np.ndarray]:
    """Distance from (px, py) to the obstacle outline, and the outward normal."""
    kind, ox, oy, hx, hy = obstacle
    dx, dy = px - ox, py - oy
    if kind == 0:
        r = np.hypot(dx, dy)
        return r - hx, np.array([dx, dy]) / max(r, 1e-9)
    qx, qy = abs(dx) - hx, abs(dy) - hy
    if qx <= 0 and qy <= 0:  # inside: push out through the nearer face
        return max(qx, qy), (np.array([np.sign(dx), 0.0]) if qx > qy else np.array([0.0, np.sign(dy)]))
    v = np.array([max(qx, 0.0) * np.sign(dx), max(qy, 0.0) * np.sign(dy)])
    d = np.linalg.norm(v)
    return d, v / d


class ScriptedExpert:
    def __init__(self, cfg, seed: int | None = None):
        x = cfg.expert
        self.limits = cfg.action.limits
        self.dt = clocks_from_cfg(cfg).dt_control
        self.half_width = float(cfg.env.corridor.width_m) / 2
        self.speed = float(x.speed)
        self.lookahead = float(x.lookahead_m)
        self.clearance = float(x.clearance_m)
        self.wall_margin = float(x.wall_margin_m)
        self.influence = float(x.influence_m)
        self.repulsion = float(x.repulsion)
        self.max_heading = np.deg2rad(float(x.max_heading_deg))
        self.kp, self.ki, self.kd = (float(g) for g in x.heading_pid)
        self.altitude_gain = float(x.altitude_gain)
        self.wander_sd = np.deg2rad(float(x.wander_deg))
        self.wander_tau = float(x.wander_tau_s)
        self.rng = np.random.default_rng(seed)

    def reset(self, info: dict) -> None:
        self.obstacles = np.asarray(info["obstacles"], dtype=float)
        self.z_target = float(info["pos"][-1][2])
        self.integral = 0.0
        self.wander = 0.0

    def goal(self, x: float) -> tuple[float, float]:
        """Attractive goal as (distance ahead, y)."""
        ahead = [ob for ob in self.obstacles if -ob[3] < ob[1] - x < self.lookahead]
        if not ahead:
            return self.lookahead, 0.0
        _, ox, oy, _, hy = min(ahead, key=lambda ob: ob[1])
        lo, hi = oy - hy - self.clearance, oy + hy + self.clearance
        if not lo < 0.0 < hi:
            return self.lookahead, 0.0
        wall = self.half_width - self.wall_margin
        left, right = (hi, wall), (-wall, lo)
        gap = left if left[1] - left[0] >= right[1] - right[0] else right
        return max(ox - x, 0.5), 0.5 * (gap[0] + gap[1])

    def field(self, x: float, y: float) -> tuple[np.ndarray, float]:
        """Field vector and the distance to the nearest obstacle surface."""
        dx, gy = self.goal(x)
        f = np.array([dx, gy - y])
        f /= np.linalg.norm(f)
        nearest = np.inf
        for ob in self.obstacles:
            if ob[1] - x < -ob[3] - self.influence:
                continue
            d, n = surface_distance(x, y, ob)
            nearest = min(nearest, d)
            if d < self.influence:
                f += self.repulsion * (1.0 / max(d, 0.05) - 1.0 / self.influence) * n
        for d, n in ((self.half_width - y, np.array([0.0, -1.0])), (self.half_width + y, np.array([0.0, 1.0]))):
            if d < self.influence:
                f += self.repulsion * (1.0 / max(d, 0.05) - 1.0 / self.influence) * n
        return f, nearest

    def act(self, info: dict) -> np.ndarray:
        x, y, z = info["pos"][-1]
        rel_yaw = float(info["rel_yaw"][-1])
        yaw = float(info["yaw"][-1])
        f, nearest = self.field(x, y)

        a = self.dt / self.wander_tau
        self.wander += -a * self.wander + self.wander_sd * np.sqrt(2 * a) * self.rng.standard_normal()
        heading = np.clip(np.arctan2(f[1], f[0]) + self.wander, -self.max_heading, self.max_heading)

        err = float(wrap(heading - rel_yaw))
        self.integral = np.clip(self.integral + err * self.dt, -0.5, 0.5)
        yaw_rate = self.kp * err + self.ki * self.integral - self.kd * float(info["yaw_rate"][-1])

        speed = self.speed * np.clip((nearest - 0.2) / 0.8, 0.4, 1.0)
        world_heading = heading + (yaw - rel_yaw)
        vx, vy = speed * np.cos(world_heading), speed * np.sin(world_heading)
        c, s = np.cos(yaw), np.sin(yaw)
        cmd = [c * vx + s * vy, -s * vx + c * vy, self.altitude_gain * (self.z_target - z), yaw_rate]
        return normalize_action(cmd, self.limits).astype(np.float32)
