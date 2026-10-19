"""Behavior cloning from the T4/T5 cache (never runs flyvis).

* Whole episodes from step 0 (heading is relative to episode start, so the core must integrate
  from t = 0); episodes in a batch are zero-padded and the padding is masked out of the loss.
* Truncated BPTT over windows of ``bc.tbptt_core_steps`` core steps: state carried across
  windows, detached at each boundary; one optimizer update per window.
* Loss = MSE(actor mean, expert action) at decision steps (``Clocks``)
  + ``aux_weight`` * MSE(heading decoder, (sin, cos)(rel_yaw)) at every core step.
* Adam (``bc.lr``), grad-norm clipping (``bc.grad_clip``), ``bc.batch_episodes`` episodes per batch.
* The validation episodes are scored after every epoch; the epoch with the best validation heading
  R^2 is also saved as ``model_best.pt``.
* fp32 compute: the fp16 cache is upcast per window.
* ``bc.pauses`` (off when absent) splices synthetic pauses into the training episodes
  (``flybywire/data/pause_bank.py``, D39); validation and test episodes are never spliced.

CLI::

    python -m flybywire.train.bc --cache data/cache --out runs/bc/real_s0 [key=value ...]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flybywire.clocks import Clocks, clocks_from_cfg
from flybywire.data.cache import T4T5Cache
from flybywire.data.pause_bank import PauseBank, PauseSampler, splice
from flybywire.data.trajectories import Trajectory, heading_target
from flybywire.eval.metrics import angular_error_deg, heading_r2


@dataclass
class EpisodeTargets:
    actions: np.ndarray  # (K, 4)
    heading: np.ndarray  # (T, 2)


class BCData:
    """Cache-backed episodes plus their BC targets."""

    def __init__(self, cache: T4T5Cache, clocks: Clocks, pauses: PauseSampler | None = None):
        self.cache = cache
        self.clocks = clocks
        self.pauses = pauses
        if pauses is not None:
            pauses.bank.check_cache(cache)
        self.targets: list[EpisodeTargets] = []
        for i, p in enumerate(cache.trajectories):
            tr = Trajectory.load(p, load_hexals=False)
            if len(tr.rel_yaw) != cache.length(i):
                raise ValueError(f"{p}: {len(tr.rel_yaw)} steps but cache has {cache.length(i)}")
            if clocks.n_core_steps(tr.n_decisions) != cache.length(i):
                raise ValueError(f"{p}: decision/core-step alignment mismatch")
            self.targets.append(
                EpisodeTargets(tr.actions.astype(np.float32), heading_target(tr.rel_yaw).astype(np.float32))
            )

    def __len__(self) -> int:
        return len(self.targets)

    def episode(self, i, rng=None):
        """Frames, heading targets, actions and action mask of episode i, with pauses if ``rng`` is given."""
        frames, heading, actions = self.cache.episode(i), self.targets[i].heading, self.targets[i].actions
        entries = self.pauses.draw(i, rng) if self.pauses is not None and rng is not None else []
        if len(entries) == 0:
            return frames, heading, actions, np.ones(len(actions), dtype=bool)
        return splice(frames, heading, actions, self.pauses.bank, entries, self.clocks)

    def batch(self, idx, device, rng=None) -> dict[str, torch.Tensor]:
        spd = self.clocks.sensory_per_decision
        eps = [self.episode(i, rng) for i in idx]
        lengths = [len(e[0]) for e in eps]
        tmax = int(math.ceil(max(lengths) / spd) * spd)
        kmax = tmax // spd
        b = len(idx)
        x = np.zeros((b, tmax, *self.cache.data.shape[1:]), dtype=np.float16)
        head = np.zeros((b, tmax, 2), dtype=np.float32)
        mask = np.zeros((b, tmax), dtype=bool)
        act = np.zeros((b, kmax, 4), dtype=np.float32)
        amask = np.zeros((b, kmax), dtype=bool)
        for j, (frames, heading, actions, action_mask) in enumerate(eps):
            t, k = lengths[j], len(actions)
            x[j, :t] = frames
            head[j, :t] = heading
            mask[j, :t] = True
            act[j, :k] = actions
            amask[j, :k] = action_mask
        to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
        return {"x16": to(x), "heading": to(head), "mask": to(mask), "actions": to(act), "amask": to(amask)}


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(pred.dtype).unsqueeze(-1)
    denom = m.sum() * pred.shape[-1]
    return ((pred - target) ** 2 * m).sum() / denom.clamp_min(1.0)


def window_loss(model, batch, v, s, e, aux_weight, clocks):
    out = model.run(batch["x16"][:, s:e].float(), v, t0=s)
    lh = _masked_mse(out["heading"], batch["heading"][:, s:e], batch["mask"][:, s:e])
    k = (s + out["decision_t"]) // clocks.sensory_per_decision
    la = _masked_mse(out["action_mean"], batch["actions"][:, k], batch["amask"][:, k])
    return la + aux_weight * lh, la, lh, out


@torch.no_grad()
def evaluate(model, data: BCData, idx, device, batch_episodes: int = 32) -> dict[str, float]:
    model.eval()
    preds, targets, masks, aerr = [], [], [], []
    for start in range(0, len(idx), batch_episodes):
        sub = idx[start : start + batch_episodes]
        batch = data.batch(sub, device)
        v = model.init_state(len(sub))
        out = model.run(batch["x16"].float(), v, t0=0)
        preds.append(out["heading"].cpu())
        targets.append(batch["heading"].cpu())
        masks.append(batch["mask"].cpu())
        k = out["decision_t"] // data.clocks.sensory_per_decision
        am = batch["amask"][:, k].unsqueeze(-1).float()
        se = ((out["action_mean"] - batch["actions"][:, k]) ** 2) * am
        aerr.append(float(se.sum() / (am.sum() * 4).clamp_min(1)))
    model.train()
    p = torch.cat([q.reshape(-1, 2) for q in preds])
    t = torch.cat([q.reshape(-1, 2) for q in targets])
    m = torch.cat([q.reshape(-1) for q in masks])
    return {
        "heading_r2": heading_r2(p, t, m),
        "heading_err_deg": angular_error_deg(p, t, m),
        "action_mse": float(np.mean(aerr)),
    }


def train_bc(
    cfg,
    cache_dir: str | Path,
    out_dir: str | Path,
    n_updates: int | None = None,
    device: str | None = None,
    val_fraction: float = 0.1,
    log_every: int = 25,
    use_wandb: bool = False,
    verbose: bool = True,
    conn=None,
    lptc=None,
) -> dict:
    """Train BC; ``conn`` / ``lptc`` override the config-built connectome / LPTC (tests, smoke)."""
    from flybywire.core.model import build_model

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    clocks = clocks_from_cfg(cfg)

    cache = T4T5Cache(cache_dir)
    cache.check_compatible(cfg.vision.flyvis_model, clocks.dt_sensory)
    pcfg = OmegaConf.select(cfg, "bc.pauses", default=None)
    sampler = PauseSampler(PauseBank(pcfg.bank), int(pcfg.per_episode), float(pcfg.p)) if pcfg else None
    data = BCData(cache, clocks, sampler)
    rng = np.random.default_rng(int(cfg.seed))
    pause_rng = np.random.default_rng([int(cfg.seed), 1])  # separate, so the epoch order matches a run without pauses
    order = rng.permutation(len(data))
    n_val = int(round(val_fraction * len(data))) if len(data) >= 10 else 0
    val_idx, train_idx = order[:n_val].tolist(), order[n_val:].tolist()

    model = build_model(cfg, conn=conn, lptc=lptc, verbose=verbose).to(device)
    params = [p for n, p in model.named_parameters() if not n.startswith("critic.") and n != "actor.log_std"]
    opt = torch.optim.Adam(params, lr=float(cfg.bc.lr))
    bcfg = cfg.bc
    aux = float(bcfg.aux_weight)
    max_updates = n_updates if n_updates is not None else 10**12
    max_epochs = int(bcfg.epochs) if n_updates is None else 10**9

    run = None
    if use_wandb and cfg.logging.wandb_mode != "disabled":
        import wandb

        run = wandb.init(project="flybywire", mode=cfg.logging.wandb_mode, dir=str(out_dir),
                         config=OmegaConf.to_container(cfg, resolve=True))

    def save(name: str) -> None:
        torch.save({"model": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True)}, out_dir / name)

    history = []
    best = {"epoch": None, "heading_r2": -math.inf}
    update = 0
    t_start = time.time()
    for epoch in range(max_epochs):
        perm = rng.permutation(train_idx).tolist()
        for start in range(0, len(perm), int(bcfg.batch_episodes)):
            idx = perm[start : start + int(bcfg.batch_episodes)]
            batch = data.batch(idx, device, pause_rng)
            v = model.init_state(len(idx))
            for s, e in clocks.windows(batch["x16"].shape[1], int(bcfg.tbptt_core_steps)):
                if not bool(batch["mask"][:, s:e].any()):
                    break
                loss, la, lh, out = window_loss(model, batch, v, s, e, aux, clocks)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(params, float(bcfg.grad_clip))
                opt.step()
                v = out["v"].detach()
                rec = {"update": update, "epoch": epoch, "loss": loss.item(), "action_mse": la.item(),
                       "heading_mse": lh.item(), "grad_norm": float(gnorm), "window_start": s}
                if not all(math.isfinite(rec[k]) for k in ("loss", "action_mse", "heading_mse")):
                    raise FloatingPointError(f"non-finite loss at update {update}: {rec}")
                history.append(rec)
                if run is not None:
                    run.log(rec, step=update)
                if verbose and update % log_every == 0:
                    print(f"[bc] upd {update:5d} ep {epoch:3d} loss {rec['loss']:.4f} "
                          f"act {rec['action_mse']:.4f} head {rec['heading_mse']:.4f} |g| {rec['grad_norm']:.3f}")
                update += 1
                if update >= max_updates:
                    break
            if update >= max_updates:
                break
        if val_idx:
            val = evaluate(model, data, val_idx, device, int(bcfg.batch_episodes))
            history[-1].update({f"val_{k}": v for k, v in val.items()})
            if val["heading_r2"] > best["heading_r2"]:
                best = {"epoch": epoch, "heading_r2": val["heading_r2"]}
                save("model_best.pt")
            if run is not None:
                run.log({f"val_{k}": v for k, v in val.items()}, step=update)
            if verbose:
                print(f"[bc] epoch {epoch:3d} val heading R2 {val['heading_r2']:.3f} "
                      f"err {val['heading_err_deg']:.2f} deg act {val['action_mse']:.4f} (best epoch {best['epoch']})")
        if update >= max_updates:
            break

    eval_idx = val_idx if val_idx else train_idx
    metrics = {
        "split": "val" if val_idx else "train",
        **evaluate(model, data, eval_idx, device, int(bcfg.batch_episodes)),
        "best_epoch": best["epoch"],
        "updates": update,
        "seconds": round(time.time() - t_start, 1),
        "mask_type": cfg.connectome.mask_type,
        "source": cfg.connectome.source,
        "seed": int(cfg.seed),
        "cache": str(Path(cache_dir).resolve()),
        **({"pauses": {"bank": str(sampler.bank.dir.resolve()), "per_episode": sampler.per_episode, "p": sampler.p}}
           if sampler else {}),
    }
    save("model.pt")
    (out_dir / "history.json").write_text(json.dumps(history))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    OmegaConf.save(cfg, out_dir / "config.yaml")
    if run is not None:
        run.summary.update(metrics)
        run.finish()
    if verbose:
        print(f"[bc] done: {json.dumps(metrics)}")
    return {"history": history, "metrics": metrics, "model": model}


def main(argv=None) -> int:
    from flybywire.config import data_dir, load_config

    ap = argparse.ArgumentParser(description="Behavior cloning from the T4/T5 cache")
    ap.add_argument("--cache", default=None, help="cache dir (default: <data_dir>/cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--updates", type=int, default=None, help="stop after N optimizer updates")
    ap.add_argument("--device", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    cache = args.cache or str(data_dir(cfg) / "cache")
    train_bc(cfg, cache, args.out, n_updates=args.updates, device=args.device, use_wandb=args.wandb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
