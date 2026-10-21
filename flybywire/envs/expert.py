"""Scripted expert that generates the BC data. It reads the privileged state in info.

A potential field in the corridor plane pulls toward a point straight down the corridor, or toward
the widest gap in the next row of obstacles when that row blocks the centre line, and pushes away
from obstacle and wall surfaces closer than influence_m. The field direction plus a slow random
wander (seeded OU process, so dodges differ between episodes) is the desired heading, and a
heading PID turns toward it. Velocity is `speed` along that heading, altitude is held.

With `expert.maneuvers` set, it also turns on the spot now and then: slows down, yaws through a
large angle, holds, and comes back to the corridor heading, by a full rotation if that is the
shorter way. With maneuvers.targets = uniform the far heading is drawn from the whole circle
instead, reached and left the short way, so the held headings cover the circle evenly rather than
a few multiples of 45 deg (D42). The yaw reference follows a trapezoidal rate profile and replaces the heading target
(no max_heading clip) while the turn lasts. The potential field is in world coordinates, so
avoidance keeps working at any heading, within the forward-only body speed limits. Without the
key the expert is the plain one.

It acts on the last frame of each window, which is what decision k sees.
"""

from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from flybywire.clocks import clocks_from_cfg
from flybywire.envs.corridor import normalize_action, wrap

ROW_TOL = 0.3  # m along x; obstacles closer than this belong to one row (other levels keep >= 1.1 m)


def surface_distance(px: float, py: float, obstacle) -> tuple[float, np.ndarray]:
    """Distance from (px, py) to the obstacle outline, and the outward normal."""
    kind, ox, oy, hx, hy = obstacle
    dx, dy = px - ox, py - oy
    if kind == 0:
        r = np.hypot(dx, dy)
        return r - hx, np.array([dx, dy]) / max(r, 1e-9)
    qx, qy = abs(dx) - hx, abs(dy) - hy
    if qx <= 0 and qy <= 0:
        return max(qx, qy), (np.array([np.sign(dx), 0.0]) if qx > qy else np.array([0.0, np.sign(dy)]))
    v = np.array([max(qx, 0.0) * np.sign(dx), max(qy, 0.0) * np.sign(dy)])
    d = np.linalg.norm(v)
    return d, v / d


def turn_profile(angle: float, peak: float, accel: float, t: float) -> tuple[float, float, float]:
    """Angle turned and rate at time t of a rest-to-rest turn of `angle` > 0, plus its duration."""
    peak = min(peak, np.sqrt(angle * accel))
    t_acc = peak / accel
    t_end = 2 * t_acc + (angle - peak * t_acc) / peak
    if t >= t_end:
        return angle, 0.0, t_end
    if t < t_acc:
        return 0.5 * accel * t * t, accel * t, t_end
    if t < t_end - t_acc:
        return 0.5 * peak * t_acc + peak * (t - t_acc), peak, t_end
    return angle - 0.5 * accel * (t_end - t) ** 2, accel * (t_end - t), t_end


class ExecNoise:
    """OU noise added to the action that is executed, while the label stays the expert's own (DART).

    The states it leads to are ones the clean expert never visits, each labelled with what the
    expert does there. ``spec`` is expert.exec_noise: sd per action dim (normalised units), tau_s,
    and the range the per-episode multiplier of sd is drawn from, so some episodes stay nearly clean.
    """

    def __init__(self, spec, dt: float, seed: int):
        self.rng = np.random.default_rng([seed, 2])
        self.sd = np.asarray(spec.sd, dtype=float) * self.rng.uniform(*spec.scale)
        self.rate = dt / float(spec.tau_s)
        self.x = np.zeros(len(self.sd))

    def __call__(self, action: np.ndarray) -> np.ndarray:
        self.x += -self.rate * self.x + self.sd * np.sqrt(2 * self.rate) * self.rng.standard_normal(len(self.sd))
        return np.clip(action + self.x, -1.0, 1.0).astype(np.float32)


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
        self.turns = OmegaConf.select(cfg, "expert.maneuvers", default=None)
        self.turn_rng = np.random.default_rng(None if seed is None else [seed, 1])

    def reset(self, info: dict) -> None:
        self.obstacles = np.asarray(info["obstacles"], dtype=float)
        self.z_target = float(info["pos"][-1][2])
        self.integral = 0.0
        self.wander = 0.0
        if self.turns is not None:
            self.t = 0.0
            self.yaw_u = float(info["rel_yaw"][-1])
            self.legs = []
            self.next_turn = self.turn_rng.uniform(*self.turns.wait_s)

    def goal(self, x: float) -> tuple[float, float]:
        """Attractive goal as (distance ahead, y)."""
        ahead = [ob for ob in self.obstacles if -ob[3] < ob[1] - x < self.lookahead]
        if not ahead:
            return self.lookahead, 0.0
        ox = min(ob[1] for ob in ahead)
        # every pillar of the nearest row, passed or not, so the chosen gap holds until the row is
        # behind; rows of two exist only on the forest level (D46), elsewhere a row is one obstacle
        row = sorted((ob[2] - ob[4] - self.clearance, ob[2] + ob[4] + self.clearance)
                     for ob in self.obstacles if abs(ob[1] - ox) < ROW_TOL)
        if not any(lo < 0.0 < hi for lo, hi in row):
            return self.lookahead, 0.0
        wall = self.half_width - self.wall_margin
        edges = [-wall, *(e for span in row for e in span), wall]
        gaps = list(zip(edges[::2], edges[1::2], strict=True))
        gap = max(reversed(gaps), key=lambda g: g[1] - g[0])
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

    def clear_to_turn(self, x: float, nearest: float) -> bool:
        ahead = [ob[1] - ob[3] - x for ob in self.obstacles if ob[1] + ob[3] > x]
        return nearest > self.turns.clear_m and min(ahead, default=np.inf) > self.turns.clear_ahead_m

    def start_turn(self) -> None:
        rng = self.turn_rng
        if self.turns.get("targets", "angles") == "uniform":
            home = 360.0 * np.round(np.degrees(self.yaw_u) / 360.0)
            far = home + rng.uniform(-180.0, 180.0)
        else:
            far = float(rng.choice(np.asarray(self.turns.angles_deg))) * rng.choice([-1, 1])
            home = 360.0 * np.round(far / 360.0)
        out = [(far, rng.uniform(*self.turns.hold_s))] if far != home else []
        self.legs = [(np.deg2rad(angle), hold) for angle, hold in (*out, (home, 0.0))]
        self.leg_from, self.leg_t = self.yaw_u, 0.0

    def turn(self, x: float, nearest: float, rel_yaw: float) -> tuple[float, float] | None:
        """Yaw error and rate feedforward while a maneuver runs, None otherwise."""
        self.t += self.dt
        self.yaw_u += float(wrap(rel_yaw - self.yaw_u))
        if not self.legs:
            if self.t < self.next_turn or not self.clear_to_turn(x, nearest):
                return None
            self.start_turn()
        target, hold = self.legs[0]
        delta = target - self.leg_from
        turned, rate, duration = turn_profile(abs(delta), self.turns.peak_rate, self.turns.accel, self.leg_t)
        ref = self.leg_from + np.sign(delta) * turned
        self.leg_t += self.dt
        if self.leg_t >= duration + hold:
            self.leg_from, self.leg_t = target, 0.0
            self.legs.pop(0)
            if not self.legs:
                self.next_turn = self.t + self.turn_rng.uniform(*self.turns.wait_s)
        return ref - self.yaw_u, np.sign(delta) * rate

    def act(self, info: dict) -> np.ndarray:
        x, y, z = info["pos"][-1]
        rel_yaw = float(info["rel_yaw"][-1])
        yaw = float(info["yaw"][-1])
        f, nearest = self.field(x, y)

        a = self.dt / self.wander_tau
        self.wander += -a * self.wander + self.wander_sd * np.sqrt(2 * a) * self.rng.standard_normal()
        heading = np.clip(np.arctan2(f[1], f[0]) + self.wander, -self.max_heading, self.max_heading)

        err, ff, speed = float(wrap(heading - rel_yaw)), 0.0, self.speed
        turning = self.turn(x, nearest, rel_yaw) if self.turns is not None else None
        if turning is not None:
            err, ff = turning
            speed = float(self.turns.speed)
        self.integral = np.clip(self.integral + err * self.dt, -0.5, 0.5)
        yaw_rate = ff + self.kp * err + self.ki * self.integral - self.kd * (float(info["yaw_rate"][-1]) - ff)

        speed *= np.clip((nearest - 0.2) / 0.8, 0.4, 1.0)
        world_heading = heading + (yaw - rel_yaw)
        vx, vy = speed * np.cos(world_heading), speed * np.sin(world_heading)
        c, s = np.cos(yaw), np.sin(yaw)
        cmd = [c * vx + s * vy, -s * vx + c * vy, self.altitude_gain * (self.z_target - z), yaw_rate]
        return normalize_action(cmd, self.limits).astype(np.float32)
