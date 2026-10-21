"""Corridor flight env: gym-pybullet-drones CF2X, forward FPV camera, velocity + yaw-rate actions.

The corridor runs along world +x (x in [0, length], |y| < width / 2, z in [0, height]) with
floor-to-ceiling cylinders and boxes. The drone starts at x = START_X facing +x, so yaw_0 is the
corridor axis.

Actions in [-1, 1]^4 map to [v_forward, v_lateral, v_vertical, yaw_rate] (action.limits) and are
tracked by DSLPIDControl every physics step. reset() runs the hover pre-roll and step(a) holds a
for one decision; both return 4 frames (4, H, W) uint8, timed as in flybywire/clocks.py.

info arrays have one row per frame: pos, quat, vel, ang_vel (world frame), yaw, rel_yaw,
yaw_rate (body z, rad/s) and physics_step. reset() also returns obstacles, rows of
(kind, x, y, half_x, half_y) with kind 0 = cylinder, 1 = box.

The progress reward is speed along the axis in m/s rather than distance per decision, which keeps
it on the same scale as the heading term.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import logging
import os
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pybullet as p
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gymnasium import spaces
from gymnasium.utils import seeding
from omegaconf import OmegaConf

from flybywire.clocks import clocks_from_cfg
from flybywire.config import ACTION_KEYS, data_dir, obstacle_texture

log = logging.getLogger(__name__)

START_X = 1.0
FIRST_OBSTACLE_X = 4.0
OBSTACLE_Y_MAX = 0.7
CYLINDER_RADIUS = (0.15, 0.3)
BOX_HALF_SIZE = (0.15, 0.35)
MIN_GAP = 1.1
END_MARGIN = 1.0
WALL_SEGMENT = 2.5
CAMERA_OFFSET = 0.03
NEAR, FAR = 0.05, 40.0
GRAY = np.array([0.299, 0.587, 0.114])
TEXTURED_OBSTACLE_GRAY = 0.6


def action_bounds(limits) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([float(limits[k][0]) for k in ACTION_KEYS])
    hi = np.array([float(limits[k][1]) for k in ACTION_KEYS])
    return lo, hi


def denormalize_action(a: np.ndarray, limits) -> np.ndarray:
    lo, hi = action_bounds(limits)
    return lo + (np.clip(a, -1.0, 1.0) + 1.0) * 0.5 * (hi - lo)


def normalize_action(v: np.ndarray, limits) -> np.ndarray:
    lo, hi = action_bounds(limits)
    return np.clip(2.0 * (np.asarray(v, dtype=float) - lo) / (hi - lo) - 1.0, -1.0, 1.0)


def body_to_world_velocity(v_body: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy, vz = v_body
    return np.array([c * vx - s * vy, s * vx + c * vy, vz])


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _texture_file(kind: str, directory: Path, size: int = 256) -> str:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}_{size}.png"
    if path.exists():
        return str(path)
    rng = np.random.default_rng(0)
    if kind == "plain":
        img = np.full((size, size), 255, np.uint8)
    elif kind == "checker":
        idx = np.arange(size) * 8 // size
        img = np.where((idx[:, None] + idx[None, :]) % 2 == 0, 210, 60).astype(np.uint8)
    else:
        coarse = np.kron(rng.uniform(0, 255, (size // 8, size // 8)), np.ones((8, 8)))
        img = (0.6 * coarse + 0.4 * rng.uniform(0, 255, (size, size))).astype(np.uint8)
    # parallel workers may write the same file
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".png")
    os.close(fd)
    Image.fromarray(img).convert("RGB").save(tmp)
    os.replace(tmp, path)
    return str(path)


class _Aviary(BaseAviary):
    def __init__(self, env: CorridorEnv, physics_hz: int, gui: bool):
        self._env = env
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            physics=Physics.PYB,
            pyb_freq=physics_hz,
            ctrl_freq=physics_hz,
            gui=gui,
            obstacles=True,
            user_debug_gui=False,
        )

    def _addObstacles(self):
        self._env._build_corridor(self.CLIENT)

    def _actionSpace(self):
        return spaces.Box(0.0, self.MAX_RPM, (1, 4), np.float32)

    def _observationSpace(self):
        return spaces.Box(-np.inf, np.inf, (1, 0), np.float32)

    def _computeObs(self):
        return None

    def _preprocessAction(self, action):
        return np.clip(action, 0.0, self.MAX_RPM)

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        return False

    def _computeInfo(self):
        return {}


class CorridorEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, cfg, seed: int | None = None, render_mode: str | None = None, gui: bool = False):
        e = cfg.env
        self.clocks = clocks_from_cfg(cfg)
        self.size = int(e.frame_size)
        self.fov = float(e.camera_fov_deg)
        self.tilt = np.deg2rad(float(e.camera_tilt_deg))
        self.length = float(e.corridor.length_m)
        self.width = float(e.corridor.width_m)
        self.height = float(e.corridor.height_m)
        self.n_obstacles = int(e.n_obstacles)
        spacing = OmegaConf.select(cfg, "env.obstacle_spacing_m", default=None)
        self.spacing = None if spacing is None else float(spacing)
        self.pair_prob = float(OmegaConf.select(cfg, "env.pair_prob", default=0.0))
        self.texture = str(e.wall_texture)
        self.obstacle_texture = obstacle_texture(cfg)
        self.n_decisions = self.clocks.n_decisions(float(e.episode_seconds))
        self.limits = cfg.action.limits
        self.weights = {k: float(v) for k, v in cfg.reward.items()}
        self.render_mode = render_mode
        self._texture_dir = data_dir(cfg) / "textures"

        self.observation_space = spaces.Box(0, 255, (self.clocks.sensory_per_decision, self.size, self.size), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.hover_action = normalize_action(np.zeros(4), self.limits).astype(np.float32)

        self.obstacles = np.zeros((0, 5))
        # BaseAviary prints its URDF parameters every time
        with contextlib.redirect_stdout(io.StringIO()):
            self._aviary = _Aviary(self, self.clocks.physics_hz, gui)
            self._ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self._cid = self._aviary.CLIENT
        self._drone = int(self._aviary.DRONE_IDS[0])
        self.renderer, self._renderer_flag = self._pick_renderer(str(e.renderer), gui)
        log.info("CorridorEnv renderer: %s", self.renderer)
        self._proj = p.computeProjectionMatrixFOV(self.fov, 1.0, NEAR, FAR, physicsClientId=self._cid)
        self._last_frame = np.zeros((self.size, self.size), np.uint8)
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

    def _pick_renderer(self, choice: str, gui: bool) -> tuple[str, int]:
        if gui:
            return "gui", p.ER_BULLET_HARDWARE_OPENGL
        if choice in ("auto", "egl"):
            spec = importlib.util.find_spec("eglRenderer")
            plugin = p.loadPlugin(spec.origin, "_eglRendererPlugin", physicsClientId=self._cid) if spec else -1
            if plugin >= 0:
                return "egl", p.ER_BULLET_HARDWARE_OPENGL
            if choice == "egl":
                raise RuntimeError("env.renderer=egl but the pybullet EGL plugin did not load")
        return "tiny", p.ER_TINY_RENDERER

    def _sample_layout(self) -> np.ndarray:
        rng = self.np_random
        n = self.n_obstacles
        if self.pair_prob > 0:
            return self._sample_rows()
        if self.spacing is None:
            xs = np.linspace(FIRST_OBSTACLE_X, self.length - 2.0, n) + rng.uniform(-0.5, 0.5, n)
        else:
            # dense stage (D44): fixed spacing from the first obstacle on, jitter small enough that
            # neighbours never overlap along x
            xs = FIRST_OBSTACLE_X + self.spacing * (np.arange(n) + rng.uniform(-0.15, 0.15, n))
        rows = [self._pillar(x, rng.uniform(-OBSTACLE_Y_MAX, OBSTACLE_Y_MAX)) for x in xs]
        return np.array(rows, dtype=float).reshape(n, 5)

    def _pillar(self, x: float, y: float) -> tuple:
        rng = self.np_random
        if rng.random() < 0.5:
            r = rng.uniform(*CYLINDER_RADIUS)
            return (0, x, y, r, r)
        hx, hy = rng.uniform(*BOX_HALF_SIZE, 2)
        return (1, x, y, hx, hy)

    def _sample_rows(self) -> np.ndarray:
        # forest level (D46): a row every `spacing` m holding one pillar or, with probability
        # pair_prob, two side by side anywhere across the corridor, as long as one gap stays open
        rng = self.np_random
        half = self.width / 2
        rows, k = [], 0
        while len(rows) < self.n_obstacles:
            x = FIRST_OBSTACLE_X + self.spacing * (k + rng.uniform(-0.15, 0.15))
            k += 1
            if self.n_obstacles - len(rows) < 2 or rng.random() >= self.pair_prob:
                rows.append(self._pillar(x, rng.uniform(-OBSTACLE_Y_MAX, OBSTACLE_Y_MAX)))
                continue
            while True:
                pair = sorted((self._pillar(x, rng.uniform(-half + 0.2, half - 0.2)) for _ in range(2)),
                              key=lambda r: r[2])
                (_, _, ya, _, ha), (_, _, yb, _, hb) = pair
                gaps = (ya - ha + half, (yb - hb) - (ya + ha), half - (yb + hb))
                if gaps[1] > 0 and max(gaps) >= MIN_GAP:
                    break
            rows += pair
        return np.array(rows, dtype=float)

    def _body(self, cid, col, vis, center, texture):
        body = p.createMultiBody(0, col, vis, basePosition=center, physicsClientId=cid)
        p.changeVisualShape(body, -1, textureUniqueId=texture, physicsClientId=cid)

    def _box(self, cid, center, half, gray, texture):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[gray, gray, gray, 1], physicsClientId=cid)
        self._body(cid, col, vis, center, texture)

    def _build_corridor(self, cid) -> None:
        L, W, H = self.length, self.width, self.height
        t = 0.05
        # OpenGL puts a fine default checker on primitives that aliases at a distance, so every
        # body gets an explicit texture
        plain = p.loadTexture(_texture_file("plain", self._texture_dir), physicsClientId=cid)
        wall_tex, wall_gray = plain, 0.75
        if self.texture != "none":
            wall_tex = p.loadTexture(_texture_file(self.texture, self._texture_dir), physicsClientId=cid)
            wall_gray = 1.0
        n_seg = int(np.ceil(L / WALL_SEGMENT))
        seg = L / n_seg
        for side in (-1, 1):
            for i in range(n_seg):
                self._box(cid, [(i + 0.5) * seg, side * (W / 2 + t), H / 2], [seg / 2, t, H / 2], wall_gray, wall_tex)
        self._box(cid, [L / 2, 0, H + t], [L / 2, W / 2 + 2 * t, t], 0.55, plain)
        for x in (-t, L + t):
            self._box(cid, [x, 0, H / 2], [t, W / 2 + 2 * t, H / 2], 0.4, plain)
        cyl_gray, box_gray, obstacle_tex = 0.25, 0.35, plain
        if self.obstacle_texture != "none":
            obstacle_tex = p.loadTexture(_texture_file(self.obstacle_texture, self._texture_dir), physicsClientId=cid)
            cyl_gray = box_gray = TEXTURED_OBSTACLE_GRAY
        for kind, x, y, hx, hy in self.obstacles:
            if kind == 0:
                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=hx, height=H, physicsClientId=cid)
                vis = p.createVisualShape(p.GEOM_CYLINDER, radius=hx, length=H, rgbaColor=[cyl_gray] * 3 + [1],
                                          physicsClientId=cid)
                self._body(cid, col, vis, [x, y, H / 2], obstacle_tex)
            else:
                self._box(cid, [x, y, H / 2], [hx, hy, H / 2], box_gray, obstacle_tex)

    def _state(self) -> dict:
        av = self._aviary
        quat = av.quat[0].copy()
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        ang_vel = av.ang_v[0].copy()
        return {
            "pos": av.pos[0].copy(),
            "quat": quat,
            "vel": av.vel[0].copy(),
            "ang_vel": ang_vel,
            "rpy": av.rpy[0].copy(),
            "yaw_rate": float((rot.T @ ang_vel)[2]),
        }

    def _physics_step(self, cmd: np.ndarray) -> None:
        av = self._aviary
        dt = self.clocks.dt_physics
        pos, quat, rpy, vel = av.pos[0], av.quat[0], av.rpy[0], av.vel[0]
        self._target_yaw = float(wrap(self._target_yaw + cmd[3] * dt))
        # the controller's yaw D term differences against the last step, which jumps 2 pi at +/-pi
        last = self._ctrl.last_rpy
        last[2] += 2 * np.pi * np.round((rpy[2] - last[2]) / (2 * np.pi))
        rpm, _, _ = self._ctrl.computeControl(
            control_timestep=dt,
            cur_pos=pos,
            cur_quat=quat,
            cur_vel=vel,
            cur_ang_vel=av.ang_v[0],
            target_pos=pos,
            target_rpy=np.array([0.0, 0.0, self._target_yaw]),
            target_vel=body_to_world_velocity(cmd[:3], rpy[2]),
            target_rpy_rates=np.array([0.0, 0.0, cmd[3]]),
        )
        av.step(rpm[None])
        self._physics_count += 1
        if p.getContactPoints(bodyA=self._drone, physicsClientId=self._cid):
            self._collided = True

    def _run_window(self, cmd: np.ndarray):
        pps = self.clocks.physics_per_sensory
        frames = np.empty(self.observation_space.shape, np.uint8)
        states = []
        for f in range(self.clocks.sensory_per_decision):
            for _ in range(pps):
                self._physics_step(cmd)
            frames[f] = self.render_frame()
            s = self._state()
            s["physics_step"] = self._physics_count - 1
            states.append(s)
        return frames, states

    def _info(self, states: list[dict]) -> dict:
        yaw = np.array([s["rpy"][2] for s in states])
        return {
            "pos": np.stack([s["pos"] for s in states]),
            "quat": np.stack([s["quat"] for s in states]),
            "vel": np.stack([s["vel"] for s in states]),
            "ang_vel": np.stack([s["ang_vel"] for s in states]),
            "yaw": yaw,
            "rel_yaw": wrap(yaw - self.yaw_0),
            "yaw_rate": np.array([s["yaw_rate"] for s in states]),
            "physics_step": np.array([s["physics_step"] for s in states]),
            "yaw_0": self.yaw_0,
            "collision": self._collided,
            "x_progress": float(states[-1]["pos"][0] - START_X),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.obstacles = self._sample_layout()
        av = self._aviary
        av.INIT_XYZS = np.array([[START_X, rng.uniform(-0.3, 0.3), rng.uniform(1.0, 1.4)]])
        av.INIT_RPYS = np.zeros((1, 3))
        av.reset()
        self._drone = int(av.DRONE_IDS[0])
        self._ctrl.reset()
        self.yaw_0 = float(av.rpy[0][2])
        self._target_yaw = self.yaw_0
        self._physics_count = 0
        self._collided = False
        self._decision = 0
        self._a_prev = self.hover_action.copy()
        frames, states = self._run_window(np.zeros(4))
        info = self._info(states)
        info["obstacles"] = self.obstacles.copy()
        return frames, info

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)
        x0 = float(self._aviary.pos[0][0])
        frames, states = self._run_window(denormalize_action(a, self.limits))
        self._decision += 1
        info = self._info(states)
        pos, rpy = states[-1]["pos"], states[-1]["rpy"]
        out_of_bounds = (
            pos[0] < 0 or pos[0] > self.length or abs(pos[1]) > self.width / 2 or pos[2] < 0 or pos[2] > self.height
        )
        reached_end = pos[0] >= self.length - END_MARGIN
        w = self.weights
        reward = (
            w["progress"] * (pos[0] - x0) * self.clocks.control_hz
            + w["heading"] * np.cos(info["rel_yaw"][-1])
            + w["collision"] * float(self._collided)
            + w["attitude"] * (abs(rpy[0]) + abs(rpy[1]))
            + w["smoothness"] * float(np.linalg.norm(a - self._a_prev))
        )
        self._a_prev = a
        terminated = bool(self._collided or out_of_bounds or reached_end)
        truncated = bool(not terminated and self._decision >= self.n_decisions)
        info["out_of_bounds"] = bool(out_of_bounds)
        info["reached_end"] = bool(reached_end)
        return frames, float(reward), terminated, truncated, info

    def render_frame(self) -> np.ndarray:
        av = self._aviary
        rot = np.array(p.getMatrixFromQuaternion(av.quat[0])).reshape(3, 3)
        forward = rot @ np.array([np.cos(self.tilt), 0.0, -np.sin(self.tilt)])
        up = rot @ np.array([np.sin(self.tilt), 0.0, np.cos(self.tilt)])
        eye = av.pos[0] + rot @ np.array([CAMERA_OFFSET, 0.0, 0.0])
        view = p.computeViewMatrix(eye, eye + forward, up, physicsClientId=self._cid)
        _, _, rgba, _, _ = p.getCameraImage(
            self.size, self.size, view, self._proj,
            shadow=0, renderer=self._renderer_flag, flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self._cid,
        )
        rgb = np.asarray(rgba, dtype=np.uint8).reshape(self.size, self.size, 4)[..., :3]
        self._last_frame = np.round(rgb @ GRAY).astype(np.uint8)
        return self._last_frame

    def render(self):
        return np.repeat(self._last_frame[..., None], 3, axis=-1)

    def close(self) -> None:
        if p.isConnected(self._cid):
            self._aviary.close()
