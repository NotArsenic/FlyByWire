"""PPO fine-tuning from a BC checkpoint. Written by hand, no RLlib or SB3.

Rollouts
    Synchronous, on-policy. ``ppo.n_envs`` envs in ``AsyncVectorEnv(shared_memory=True,
    autoreset_mode=SAME_STEP)`` (``flybywire/envs/vec.py``) only render and step physics; BoxEye,
    flyvis and the core run batched here, as in closed-loop evaluation. A rollout is
    ``ppo.rollout_decisions`` decisions per env. Core and flyvis state carry over from one rollout
    to the next, so episodes span rollouts.

Recurrence
    The rollout keeps the T4/T5 of every core step on the GPU and the core state at its start;
    the update reruns the core over the whole rollout from that state, cutting gradients every
    ``ppo.tbptt_core_steps``. flyvis is not rerun: it has no parameters to train.

Episode boundaries
    When an env's episode ends, the frames returned with it already belong to the next episode
    (SAME_STEP), so its core and flyvis state are reset before they are used, and the core step
    where the new episode starts is marked so the rerun resets there too. A time-limit end is
    bootstrapped with the critic's value of the final window (``final_obs``); a crash is not.

Loss
    Clipped PPO on the actor, value regression on the critic (reads ``r.detach()``, its own
    optimizer), the auxiliary heading loss as in BC so the EPG code isn't traded for reward, and
    a KL anchor to the frozen BC policy. The BC copy runs its own core on the same T4/T5 during
    the rollout, so the anchor compares the two policies on the same input history. Beta falls
    linearly from ``kl_anchor_beta`` to ``kl_anchor_beta_min`` over ``kl_anchor_decay_updates``
    updates after the warm-up. For the first ``critic_warmup_iters`` updates only the critic
    learns: with an untrained critic the first policy gradients would be mostly noise.

Checkpoints
    Every ``ppo.eval_every`` updates the action mean flies ``ppo.eval_episodes`` validation seeds
    from ``ppo.eval_seed``; model_best.pt is the best of these, the BC policy included. The
    10000.. seeds of the closed-loop tests are not used for picking.

Watching
    ``--watch`` opens one live window (flybywire/train/watch.py) on env 0: a chase camera, what the
    eye sees, the corridor from above, the EPG ring and the training curves. Closing it doesn't
    touch training. ``--record-every N`` saves that window as <out>/watch/update_XXXX.mp4 for one
    episode every N updates, so the runs can be compared over training.

Usage:
  python -m flybywire.train.ppo --bc runs/sweep_dart/real_anatomical_s5/model_best.pt --out runs/ppo/demo
                                [--updates N] [--hours H] [--watch] [--record-every N] [--resume]
                                [key=value ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from flybywire.clocks import clocks_from_cfg


@dataclass
class RolloutBuffer:
    """One rollout, on the GPU. E envs, K decisions, T = 4K core steps.

    t4t5 (E, T, 8, 721), rel_yaw (E, T), episode_start (E, T) bool (reset the core before step t),
    v0 (E, N) core state before step 0, actions (E, K, 4) sampled (unclipped), logp (E, K),
    values (E, K), rewards (E, K) scaled and with time-limit bootstraps added, dones (E, K) the
    episode ended after decision k, bc_mean (E, K, 4) the BC policy's action mean, last_value (E,).
    """

    t4t5: torch.Tensor
    rel_yaw: torch.Tensor
    episode_start: torch.Tensor
    v0: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    bc_mean: torch.Tensor
    last_value: torch.Tensor


def exploration_log_std(cfg) -> float:
    """ppo.log_std, or the BC policy's own (heads.log_std_init) when unset."""
    value = OmegaConf.select(cfg, "ppo.log_std", default=None)
    return float(cfg.heads.log_std_init if value is None else value)


def compute_gae(rewards, values, dones, last_value, gamma: float = 0.99, lam: float = 0.95):
    """GAE over (E, K) tensors -> (advantages, returns). dones[:, k] cuts the trace after decision k."""
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for k in reversed(range(rewards.shape[1])):
        nxt = last_value if k == rewards.shape[1] - 1 else values[:, k + 1]
        keep = 1.0 - dones[:, k].float()
        delta = rewards[:, k] + gamma * nxt * keep - values[:, k]
        running = delta + gamma * lam * keep * running
        adv[:, k] = running
    return adv, adv + values


def kl_anchor_beta(update: int, warmup: int, beta0: float, beta_min: float, decay_updates: int) -> float:
    """KL-anchor weight: beta0 until the warm-up ends, then linearly down to beta_min."""
    frac = min(max(update - warmup, 0) / max(decay_updates, 1), 1.0)
    return beta0 + (beta_min - beta0) * frac


def gaussian_kl(mean_p, log_std_p, mean_q, log_std_q) -> torch.Tensor:
    """KL(p || q) of diagonal Gaussians, summed over the last dim."""
    var_p, var_q = (2 * log_std_p).exp(), (2 * log_std_q).exp()
    return (log_std_q - log_std_p + (var_p + (mean_p - mean_q) ** 2) / (2 * var_q) - 0.5).sum(-1)


def heading_target(rel_yaw: torch.Tensor) -> torch.Tensor:
    return torch.stack([rel_yaw.sin(), rel_yaw.cos()], dim=-1)


class Sampler:
    """Runs the envs with the current policy and fills RolloutBuffers."""

    def __init__(self, model, bc_model, cfg, vec, device, seed: int, watcher=None):
        from flybywire.vision.boxeye import BoxEye
        from flybywire.vision.flyvis_stepper import FlyvisStepper

        self.model, self.bc_model, self.vec, self.device = model, bc_model, vec, device
        self.clocks = clocks_from_cfg(cfg)
        self.n = vec.num_envs
        self.gamma = float(cfg.ppo.gamma)
        self.scale = float(cfg.ppo.reward_scale)
        self.eye = BoxEye(int(cfg.vision.boxeye.extent), int(cfg.vision.boxeye.kernel_size), device=device)
        self.stepper = FlyvisStepper(
            model=cfg.vision.flyvis_model,
            device=device,
            dt=self.clocks.dt_sensory,
            cell_types=tuple(cfg.vision.cell_types),
            steady_state_seconds=float(cfg.vision.steady_state_seconds),
        )
        self.stepper.reset(self.n)
        self.v = model.init_state(self.n)
        self.bc_v = bc_model.init_state(self.n)
        self.frames, info = vec.reset(seed=seed)
        self.rel_yaw = info["rel_yaw"]
        self.ep_return = np.zeros(self.n)
        self.ep_len = np.zeros(self.n, dtype=int)
        self.finished = []
        self.watcher = watcher
        if watcher is not None:
            watcher.episode(info["obstacles"][0])

    def _window(self, frames: np.ndarray):
        """Eye and flyvis for one decision window of every env: (E, 4, 8, 721)."""
        n, w = frames.shape[:2]
        x = torch.from_numpy(frames).to(self.device)
        # fp16 round trip: the BC data stored hexals as fp16
        hexals = self.eye(x.reshape(n * w, *x.shape[2:])).half().float().reshape(n, w, -1)
        return torch.stack([self.stepper.step(hexals[:, t]) for t in range(w)], dim=1)

    @torch.no_grad()
    def _final_values(self, final_obs, idx: np.ndarray) -> torch.Tensor:
        """Critic value after the last window of the envs in idx; other envs' state is left as it was."""
        flyvis, v = self.stepper.state.clone(), self.v.clone()
        frames = np.zeros_like(self.frames)
        frames[idx] = np.stack([final_obs[j] for j in idx])
        y = self._window(frames)
        for t in range(y.shape[1]):
            v, r = self.model.core_step(v, y[:, t])
        self.stepper.state = flyvis
        return self.model.critic(r)[torch.as_tensor(idx, device=self.device)]

    @torch.no_grad()
    def collect(self, n_decisions: int) -> RolloutBuffer:
        dev, n, w = self.device, self.n, self.clocks.sensory_per_decision
        t = n_decisions * w
        t4t5 = torch.empty(n, t, 8, 721, device=dev)
        rel_yaw = torch.empty(n, t, device=dev)
        start = torch.zeros(n, t, dtype=torch.bool, device=dev)
        buf = {k: torch.empty(n, n_decisions, *shape, device=dev)
               for k, shape in (("actions", (4,)), ("logp", ()), ("values", ()), ("rewards", ()), ("bc_mean", (4,)))}
        dones = torch.zeros(n, n_decisions, dtype=torch.bool, device=dev)
        v0 = self.v.clone()
        for k in range(n_decisions):
            y = self._window(self.frames)
            t4t5[:, k * w : (k + 1) * w] = y
            rel_yaw[:, k * w : (k + 1) * w] = torch.as_tensor(self.rel_yaw, dtype=torch.float32, device=dev)
            for i in range(w):
                self.v, r = self.model.core_step(self.v, y[:, i])
                self.bc_v, r_bc = self.bc_model.core_step(self.bc_v, y[:, i])
            dist = self.model.actor.distribution(r)
            a = dist.sample()
            buf["actions"][:, k] = a
            buf["logp"][:, k] = dist.log_prob(a).sum(-1)
            buf["values"][:, k] = self.model.critic(r)
            buf["bc_mean"][:, k] = self.bc_model.actor(r_bc)
            if self.watcher is not None:
                self.watcher.step(self.frames[0, -1], r[0], self.model.heading(r[0]), self.rel_yaw[0][-1])

            self.frames, reward, term, trunc, info = self.vec.step(a.cpu().numpy())
            self.ep_return += reward
            self.ep_len += 1
            reward = torch.as_tensor(reward * self.scale, dtype=torch.float32, device=dev)
            boot = np.flatnonzero(trunc & ~term)
            if len(boot):
                reward[torch.as_tensor(boot, device=dev)] += self.gamma * self._final_values(info["final_obs"], boot)
            buf["rewards"][:, k] = reward
            done = np.flatnonzero(term | trunc)
            if len(done):
                dones[torch.as_tensor(done, device=dev), k] = True
                self._end_episodes(done, info)
                if k + 1 < n_decisions:
                    start[torch.as_tensor(done, device=dev), (k + 1) * w] = True
            self.rel_yaw = info["rel_yaw"]
            if self.watcher is not None:
                self.watcher.position(info["pos"][0][-1], info["quat"][0][-1])
        return RolloutBuffer(t4t5, rel_yaw, start, v0, buf["actions"], buf["logp"], buf["values"], buf["rewards"],
                             dones, buf["bc_mean"], self._last_value())

    def _end_episodes(self, done: np.ndarray, info: dict) -> None:
        final = info["final_info"]
        for j in done:
            reason = ("collision" if final["collision"][j] else "end" if final["reached_end"][j]
                      else "out_of_bounds" if final["out_of_bounds"][j] else "time")
            self.finished.append({"return": float(self.ep_return[j]), "length": int(self.ep_len[j]),
                                  "success": reason in ("time", "end"), "reason": reason,
                                  "x_progress": float(final["x_progress"][j])})
            self.ep_return[j], self.ep_len[j] = 0.0, 0
        idx = torch.as_tensor(done, device=self.device)
        self.v[idx] = 0
        self.bc_v[idx] = 0
        self.stepper.reset(env_ids=done)
        if self.watcher is not None and 0 in done:
            self.watcher.episode(info["obstacles"][0])

    @torch.no_grad()
    def _last_value(self) -> torch.Tensor:
        """Value of the state the next rollout starts from (one window ahead, then put back)."""
        flyvis, v = self.stepper.state.clone(), self.v.clone()
        y = self._window(self.frames)
        for t in range(y.shape[1]):
            v, r = self.model.core_step(v, y[:, t])
        self.stepper.state = flyvis
        return self.model.critic(r)


def rerun(model, buf: RolloutBuffer, idx, tbptt: int, clocks):
    """Core over the rollout for envs idx from the stored start state. Returns r (B, T, N), r at decisions."""
    x = buf.t4t5[idx]
    u = model.inproj(model.sensory(x))
    v = buf.v0[idx]
    start = buf.episode_start[idx]
    rs = []
    for t in range(x.shape[1]):
        if t and t % tbptt == 0:
            v = v.detach()
        v = torch.where(start[:, t, None], torch.zeros_like(v), v)
        v, r = model.core(v, u[:, t])
        rs.append(r)
    r = torch.stack(rs, dim=1)
    dec = torch.as_tensor(clocks.decision_core_steps(buf.actions.shape[1]), device=r.device)
    return r, r.index_select(1, dec)


def ppo_update(model, buf: RolloutBuffer, opt_actor, opt_critic, cfg, update: int) -> dict:
    p = cfg.ppo
    clocks = clocks_from_cfg(cfg)
    warmup = update < int(p.critic_warmup_iters)
    beta = kl_anchor_beta(update, int(p.critic_warmup_iters), float(p.kl_anchor_beta),
                          float(p.kl_anchor_beta_min), int(p.kl_anchor_decay_updates))
    adv, ret = compute_gae(buf.rewards, buf.values, buf.dones, buf.last_value, float(p.gamma), float(p.gae_lambda))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    # the anchor is on the action mean: the BC policy gets the same exploration noise
    bc_log_std = torch.full((4,), exploration_log_std(cfg), device=adv.device)
    actor_params = [q for g in opt_actor.param_groups for q in g["params"]]
    n_env = buf.actions.shape[0]
    mb = int(p.minibatch_envs)
    logs = []
    for _ in range(int(p.epochs)):
        stop = False
        for idx in torch.randperm(n_env, device=adv.device).split(mb):
            if warmup:
                with torch.no_grad():
                    _, r_dec = rerun(model, buf, idx, int(p.tbptt_core_steps), clocks)
                v_loss = 0.5 * ((model.critic(r_dec) - ret[idx]) ** 2).mean()
                opt_critic.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.critic.parameters(), float(p.max_grad_norm))
                opt_critic.step()
                logs.append({"value_loss": v_loss.item()})
                continue
            r, r_dec = rerun(model, buf, idx, int(p.tbptt_core_steps), clocks)
            dist = model.actor.distribution(r_dec)
            logp = dist.log_prob(buf.actions[idx]).sum(-1)
            log_ratio = logp - buf.logp[idx]
            ratio = log_ratio.exp()
            a = adv[idx]
            clip = float(p.clip)
            pg_loss = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
            v_loss = 0.5 * ((model.critic(r_dec) - ret[idx]) ** 2).mean()
            kl_bc = gaussian_kl(dist.mean, model.actor.log_std, buf.bc_mean[idx], bc_log_std).mean()
            aux = ((model.heading(r) - heading_target(buf.rel_yaw[idx])) ** 2).mean()
            loss = pg_loss + beta * kl_bc + float(p.aux_weight) * aux
            opt_actor.zero_grad(set_to_none=True)
            opt_critic.zero_grad(set_to_none=True)
            (loss + v_loss).backward()
            gnorm = torch.nn.utils.clip_grad_norm_(actor_params, float(p.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), float(p.max_grad_norm))
            opt_actor.step()
            opt_critic.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean().item()
            logs.append({"pg_loss": pg_loss.item(), "value_loss": v_loss.item(), "kl_bc": kl_bc.item(),
                         "aux_heading": aux.item(), "approx_kl": approx_kl, "grad_norm": float(gnorm),
                         "clip_frac": ((ratio - 1).abs() > clip).float().mean().item()})
            if approx_kl > float(p.target_kl):
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean([g[k] for g in logs if k in g])) for k in logs[-1]}
    var = ret.var()
    out.update({"beta": beta, "warmup": warmup, "n_minibatches": len(logs),
                "explained_var": float(1 - (ret - buf.values).var() / var) if var > 0 else 0.0})
    return out


@torch.no_grad()
def validate(model, cfg) -> float:
    """Success of the action mean on ppo.eval_episodes seeds from ppo.eval_seed, which no test uses."""
    from flybywire.eval.closed_loop import PolicyDriver, fly

    p = cfg.ppo
    seeds = list(range(int(p.eval_seed), int(p.eval_seed) + int(p.eval_episodes)))
    episodes = fly(cfg, seeds, PolicyDriver(model, cfg, int(p.n_envs)), n_envs=int(p.n_envs))
    return float(np.mean([e.success for e in episodes]))


def load_bc(path, device, overrides=()):
    """Model from a BC checkpoint with the current PPO and reward settings and ``overrides`` merged in."""
    from flybywire.config import load_config
    from flybywire.core.model import build_model

    ck = torch.load(path, map_location="cpu")
    base = load_config()
    # the reward is what PPO optimises, so it comes from the current config like the PPO settings
    cfg = OmegaConf.merge(OmegaConf.create(ck["config"]), {"ppo": base.ppo, "reward": base.reward},
                          OmegaConf.from_dotlist(list(overrides)))
    model = build_model(cfg, verbose=False)
    model.load_state_dict(ck["model"])
    return model.to(device), cfg


def train_ppo(cfg, model, out_dir, n_updates: int, hours: float | None = None, device: str = "cuda",
              watch: bool = False, record_every: int | None = None, resume: bool = False,
              bc_path: str | None = None) -> dict:
    from flybywire.envs.vec import make_vec_env

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = cfg.ppo
    torch.manual_seed(int(cfg.seed))
    with torch.no_grad():
        model.actor.log_std.fill_(exploration_log_std(cfg))
    bc_model = copy.deepcopy(model).eval().requires_grad_(False)
    actor_params = [q for name, q in model.named_parameters() if not name.startswith("critic.")]
    opt_actor = torch.optim.Adam(actor_params, lr=float(p.lr_actor))
    opt_critic = torch.optim.Adam(model.critic.parameters(), lr=float(p.lr_critic))

    first, history = 0, []
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        bc_model.load_state_dict(ck["bc_model"])
        opt_actor.load_state_dict(ck["opt_actor"])
        opt_critic.load_state_dict(ck["opt_critic"])
        first = ck["update"] + 1
        # drop what was logged after the checkpoint; those updates are run again
        for name in ("history.jsonl", "validation.jsonl"):
            rows = [json.loads(line) for line in (out / name).read_text().splitlines()]
            rows = [r for r in rows if r["update"] < first]
            (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]

    def save(name: str, update: int, extra: bool = False) -> None:
        state = {"model": model.state_dict(), "config": OmegaConf.to_container(cfg, resolve=True), "update": update,
                 "bc": bc_path}
        if extra:
            state.update({"bc_model": bc_model.state_dict(), "opt_actor": opt_actor.state_dict(),
                          "opt_critic": opt_critic.state_dict()})
        torch.save(state, out / name)

    watcher = None
    if watch or record_every:
        from flybywire.eval.ring import model_layout
        from flybywire.train.watch import Watcher

        watcher = Watcher(model_layout(cfg, model, "anatomical"), model.groups[model.group_roles["heading"]], cfg)
        (out / "watch").mkdir(exist_ok=True)
    vec = make_vec_env(cfg, int(p.n_envs), seed=int(cfg.seed) * 1000 + first)
    OmegaConf.save(cfg, out / "config.yaml")
    best, t_start = -1.0, time.time()
    if first > 0:
        best = max(json.loads(line)["success"] for line in (out / "validation.jsonl").read_text().splitlines())
    else:
        best = validate(model, cfg)
        save("model_best.pt", -1)
        (out / "validation.jsonl").write_text(json.dumps({"update": -1, "success": best}) + "\n")
        print(f"[ppo] BC policy: validation success {best:.3f}", flush=True)
    try:
        sampler = Sampler(model, bc_model, cfg, vec, device, seed=int(cfg.seed) * 1000 + first, watcher=watcher)
        log = (out / "history.jsonl").open("a")
        for update in range(first, n_updates):
            t0 = time.time()
            buf = sampler.collect(int(p.rollout_decisions))
            t1 = time.time()
            stats = ppo_update(model, buf, opt_actor, opt_critic, cfg, update)
            recent = sampler.finished[-100:]
            stats.update({
                "update": update, "rollout_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1),
                "episodes": len(sampler.finished),
                "success_100": float(np.mean([e["success"] for e in recent])) if recent else math.nan,
                "return_100": float(np.mean([e["return"] for e in recent])) if recent else math.nan,
                "reward_mean": float(buf.rewards.mean() / float(p.reward_scale)),
                "log_std": model.actor.log_std.detach().cpu().tolist(),
            })
            history.append(stats)
            log.write(json.dumps(stats) + "\n")
            log.flush()
            print(f"[ppo] upd {update:4d} {'warm ' if stats['warmup'] else ''}succ {stats['success_100']:.2f} "
                  f"ret {stats['return_100']:.1f} vloss {stats['value_loss']:.3f} "
                  f"kl_bc {stats.get('kl_bc', 0):.4f} aux {stats.get('aux_heading', 0):.4f} "
                  f"ev {stats['explained_var']:.2f} ({stats['rollout_s']}+{stats['update_s']} s)", flush=True)
            if watcher is not None:
                watcher.update(stats)
                if record_every and update % record_every == 0:
                    watcher.record(out / "watch" / f"update_{update:04d}.mp4", f"update {update}")
            if (update + 1) % int(p.save_every) == 0:
                save("last.pt", update, extra=True)
                save(f"model_u{update + 1:04d}.pt", update)
            if (update + 1) % int(p.eval_every) == 0:
                val = validate(model, cfg)
                with (out / "validation.jsonl").open("a") as f:
                    f.write(json.dumps({"update": update, "success": val}) + "\n")
                print(f"[ppo] upd {update:4d} validation success {val:.3f} (best {max(best, val):.3f})", flush=True)
                if val > best:
                    best = val
                    save("model_best.pt", update)
            if hours is not None and time.time() - t_start > hours * 3600:
                break
        save("last.pt", update, extra=True)
        save("model.pt", update)
        log.close()
    finally:
        vec.close()
        if watcher is not None:
            watcher.close()
    return {"history": history, "best_validation": best}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bc", required=True, help="BC checkpoint to start from (and anchor to)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=None, help="default ppo.updates")
    ap.add_argument("--hours", type=float, default=None, help="stop after this much wall time")
    ap.add_argument("--watch", action="store_true", help="live window on env 0")
    ap.add_argument("--record-every", type=int, default=None, help="record the window every N updates")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_bc(args.bc, device, args.overrides)
    n_updates = int(args.updates or cfg.ppo.updates)
    res = train_ppo(cfg, model, args.out, n_updates, args.hours, device, args.watch, args.record_every,
                    args.resume, args.bc)
    print(f"[ppo] done, best validation success {res['best_validation']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
