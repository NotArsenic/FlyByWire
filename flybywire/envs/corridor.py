"""Corridor flight environment (STUB — implemented in week 1).

Spec: README / setup spec, section "Environment".

* gym-pybullet-drones, ``DroneModel.CF2X``; physics at ``clocks.physics_hz`` (200 Hz).
  (Verified names in the installed package: ``gym_pybullet_drones.envs.BaseAviary``,
  ``gym_pybullet_drones.control.DSLPIDControl.DSLPIDControl.computeControl(control_timestep,
  cur_pos, cur_quat, cur_vel, cur_ang_vel, target_pos, target_rpy, target_vel, target_rpy_rates)``,
  ``gym_pybullet_drones.utils.enums.DroneModel.CF2X``.)
* Corridor 30 m x 3 m x 2.5 m with ``env.n_obstacles`` cylinder / box obstacles;
  ``env.wall_texture`` in {none, checker, noise} (fallback if Gate 2 fails).
* Camera: body-fixed FPV, forward, tilted down ``env.camera_tilt_deg``, ``env.camera_fov_deg``
  (110°), ``env.frame_size`` (160) grayscale. Renderer ``auto``: try the EGL plugin
  (``eglRenderer`` / ``_eglRendererPlugin``, ``ER_BULLET_HARDWARE_OPENGL``), fall back to
  ``ER_TINY_RENDERER``; log which one is used.
* Action: normalised ``[-1, 1]^4`` -> body-frame ``[v_forward, v_lateral, v_vertical, yaw_rate]``
  (limits ``action.limits``), tracked by ``DSLPIDControl``: integrate a target yaw from
  ``yaw_rate``, rotate body velocity into a world-frame target velocity.
* Timing (``flybywire.clocks.Clocks``): one ``step(a)`` holds ``a`` for 8 physics steps and renders
  a frame after every 2 -> returns **4 frames** ``(4, H, W)`` uint8. ``reset()`` runs the 8-step
  hover pre-roll and returns its 4 frames (the inputs for decision 0).
* Observation: frames only. ``info``: pos, quat, vel, ang_vel, yaw, yaw_0, rel_yaw, yaw_rate (each
  per frame, (4, ...)), collision flag.
* Episode: ``env.episode_seconds`` (10 s = 250 decisions); terminates on collision, leaving the
  bounds, or reaching the corridor end.
* Reward: ``1.0 * forward_progress + 0.3 * cos(rel_yaw) - 10 * collision
  - 0.05 * (|roll| + |pitch|) - 0.01 * ||a - a_prev||`` (weights from ``reward``).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def denormalize_action(a: np.ndarray, limits) -> np.ndarray:
    """Map ``[-1, 1]^4`` to physical ``[v_forward, v_lateral, v_vertical, yaw_rate]`` (STUB)."""
    raise NotImplementedError("week 1: corridor env (spec 'Environment' / action space)")


def body_to_world_velocity(v_body: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate a body-frame (FLU) velocity into the world frame by ``yaw`` (STUB)."""
    raise NotImplementedError("week 1: corridor env (spec 'Environment' / action space)")


class CorridorEnv(gym.Env):
    """Single-drone obstacle corridor with FPV frames as observations (STUB)."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 100}

    def __init__(self, cfg, seed: int | None = None, render_mode: str | None = None, gui: bool = False):
        """Build the PyBullet scene, drone (CF2X), PID controller and camera from ``cfg`` (STUB)."""
        raise NotImplementedError("week 1: corridor env (spec 'Environment')")

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """New corridor layout; hover pre-roll; returns ``(frames (4, H, W) uint8, info)`` (STUB)."""
        raise NotImplementedError("week 1: corridor env (spec 'Environment')")

    def step(self, action: np.ndarray):
        """Hold ``action`` for one decision (8 physics steps, 4 frames).

        Returns ``(frames (4, H, W) uint8, reward, terminated, truncated, info)`` (STUB).
        """
        raise NotImplementedError("week 1: corridor env (spec 'Environment')")

    def render_frame(self) -> np.ndarray:
        """Render one FPV frame (H, W) uint8 with the selected renderer (STUB)."""
        raise NotImplementedError("week 1: corridor env (spec 'Environment' / camera, renderer)")

    def close(self) -> None:
        """Disconnect from PyBullet (STUB)."""
        raise NotImplementedError("week 1: corridor env")
