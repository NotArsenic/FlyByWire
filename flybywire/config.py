"""Config loading: OmegaConf load -> validation -> clock assertions.

``load_config()`` is the single entry point. It loads ``configs/default.yaml`` (or a given path),
applies dotlist overrides (``["core.phi=relu", "connectome.mask_type=rewire"]``), resolves
interpolations, validates enums / ranges / group references, and asserts that the clock rates
divide cleanly. Any problem raises :class:`ConfigError`.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from flybywire.clocks import Clocks, clocks_from_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

ENUMS = {
    "env.wall_texture": {"none", "checker", "noise"},
    "env.renderer": {"auto", "egl", "tiny"},
    "vision.backend": {"flyvis", "farneback"},
    "lptc.mode": {"frozen", "gains", "full"},
    "connectome.source": {"toy", "flywire"},
    "connectome.mask_type": {"real", "rewire", "er"},
    "core.phi": {"softplus", "relu"},
    "heads.ring_layout": {"anatomical", "tuning", "auto"},
    "action.space": {"body_vel_yawrate"},
    "logging.wandb_mode": {"online", "offline", "disabled"},
}
LPTC_CHANNELS = ["yaw", "pitch", "roll", "forward", "lateral", "vertical", "loom_left", "loom_right"]
ACTION_KEYS = ["v_forward", "v_lateral", "v_vertical", "yaw_rate"]


class ConfigError(ValueError):
    pass


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def load_config(
    path: str | os.PathLike | None = None, overrides: Sequence[str] | None = None, validate: bool = True
) -> DictConfig:
    _load_dotenv()
    cfg = OmegaConf.load(Path(path) if path else DEFAULT_CONFIG)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    if validate:
        validate_config(cfg)
    return cfg


def validate_config(cfg: DictConfig) -> Clocks:
    errors: list[str] = []

    def need(key: str):
        v = OmegaConf.select(cfg, key, default=None)
        if v is None:
            errors.append(f"missing required key: {key}")
        return v

    for key, allowed in ENUMS.items():
        v = need(key)
        if v is not None and v not in allowed:
            errors.append(f"{key}={v!r} not in {sorted(allowed)}")

    # clocks
    clocks = None
    try:
        clocks = clocks_from_cfg(cfg)
    except Exception as e:  # noqa: BLE001
        errors.append(f"clocks: {e}")
    if clocks is not None:
        try:
            clocks.n_decisions(float(cfg.env.episode_seconds))
        except Exception as e:  # noqa: BLE001
            errors.append(f"env.episode_seconds: {e}")
        tb = int(cfg.bc.tbptt_core_steps)
        if tb <= 0 or tb % clocks.sensory_per_decision:
            errors.append(f"bc.tbptt_core_steps={tb} must be a positive multiple of {clocks.sensory_per_decision}")

    # vision / lptc
    if list(cfg.lptc.channels) != LPTC_CHANNELS:
        errors.append(f"lptc.channels must be exactly {LPTC_CHANNELS}")
    if len(cfg.vision.cell_types) != 8:
        errors.append("vision.cell_types must list 8 T4/T5 subtypes")
    if int(cfg.env.frame_size) <= 0:
        errors.append("env.frame_size must be positive")

    # connectome groups
    groups = cfg.connectome.groups
    for role in ("input_group", "heading_group", "actor_group"):
        g = cfg.connectome[role]
        if g not in groups:
            errors.append(f"connectome.{role}={g!r} is not a defined group ({list(groups)})")
    for g in cfg.connectome.lesion_groups:
        if g not in groups:
            errors.append(f"connectome.lesion_groups entry {g!r} is not a defined group")
    if int(cfg.connectome.min_synapses) < 1:
        errors.append("connectome.min_synapses must be >= 1")

    # core
    a = float(cfg.core.alpha_init)
    if not 0.0 < a < 1.0:
        errors.append(f"core.alpha_init={a} must be in (0, 1)")

    # actions
    lim = cfg.action.limits
    if list(lim.keys()) != ACTION_KEYS:
        errors.append(f"action.limits keys must be {ACTION_KEYS} in this order")
    for k in lim:
        lo, hi = lim[k]
        if not lo < hi:
            errors.append(f"action.limits.{k}: low {lo} must be < high {hi}")

    # sweep
    for c in cfg.sweep.conditions:
        if c not in ENUMS["connectome.mask_type"]:
            errors.append(f"sweep.conditions entry {c!r} is not a mask type")
    if int(cfg.sweep.max_parallel) < 1:
        errors.append("sweep.max_parallel must be >= 1")

    if errors:
        raise ConfigError("invalid config:\n  - " + "\n  - ".join(errors))
    return clocks


def config_hash(cfg: DictConfig, keys: Sequence[str] | None = None) -> str:
    """Short stable hash of (a subset of) the config, for cache manifests and metadata."""
    sub = cfg if keys is None else {k: OmegaConf.select(cfg, k) for k in keys}
    blob = OmegaConf.to_yaml(OmegaConf.create(OmegaConf.to_container(sub, resolve=True)), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def data_dir(cfg: DictConfig) -> Path:
    p = Path(cfg.paths.data_dir)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def runs_dir(cfg: DictConfig) -> Path:
    p = Path(cfg.paths.runs_dir)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()
