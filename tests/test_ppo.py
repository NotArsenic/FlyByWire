"""PPO pieces that don't need the envs: GAE, KL anchor, rerunning the core over a rollout, warm-up."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flybywire.clocks import clocks_from_cfg
from flybywire.train.ppo import RolloutBuffer, compute_gae, gaussian_kl, kl_anchor_beta, ppo_update, rerun


def test_gae_without_discount_is_reward_to_go():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.0, 1.0]])
    adv, ret = compute_gae(rewards, values, torch.zeros(1, 3, dtype=torch.bool), torch.tensor([4.0]), 1.0, 1.0)
    torch.testing.assert_close(ret, torch.tensor([[10.0, 9.0, 7.0]]))
    torch.testing.assert_close(adv, ret - values)


def test_gae_stops_at_episode_end():
    rewards = torch.ones(1, 4)
    values = torch.full((1, 4), 0.5)
    dones = torch.tensor([[False, True, False, False]])
    adv, ret = compute_gae(rewards, values, dones, torch.tensor([100.0]), 0.9, 0.8)
    # decision 1 ends its episode: no bootstrap from decision 2 or from last_value
    assert ret[0, 1].item() == pytest.approx(1.0)
    assert ret[0, 0].item() == pytest.approx(1.0 + 0.9 * (0.5 + 0.8 * 0.5))


def test_kl_anchor_schedule():
    assert kl_anchor_beta(0, 10, 0.1, 0.02, 50) == pytest.approx(0.1)
    assert kl_anchor_beta(10, 10, 0.1, 0.02, 50) == pytest.approx(0.1)
    assert kl_anchor_beta(35, 10, 0.1, 0.02, 50) == pytest.approx(0.06)
    assert kl_anchor_beta(1000, 10, 0.1, 0.02, 50) == pytest.approx(0.02)


def test_gaussian_kl_matches_torch():
    g = torch.Generator().manual_seed(0)
    mp, mq = torch.randn(5, 4, generator=g), torch.randn(5, 4, generator=g)
    sp, sq = torch.randn(4, generator=g) * 0.3, torch.randn(4, generator=g) * 0.3
    ref = torch.distributions.kl_divergence(torch.distributions.Normal(mp, sp.exp()),
                                            torch.distributions.Normal(mq, sq.exp())).sum(-1)
    torch.testing.assert_close(gaussian_kl(mp, sp, mq, sq), ref)
    assert gaussian_kl(mp, sp, mp, sp).abs().max() < 1e-6


def fake_rollout(model, cfg, n_env=3, n_dec=12, reset_at=(1, 5)):
    """Run the core step by step as the sampler does, with an episode reset in env 1."""
    clocks = clocks_from_cfg(cfg)
    g = torch.Generator().manual_seed(1)
    t = n_dec * clocks.sensory_per_decision
    x = torch.rand(n_env, t, 8, 721, generator=g)
    start = torch.zeros(n_env, t, dtype=torch.bool)
    start[reset_at[0], reset_at[1] * clocks.sensory_per_decision] = True
    v = torch.rand(n_env, model.n, generator=g)
    v0, rs = v.clone(), []
    with torch.no_grad():
        for i in range(t):
            v = torch.where(start[:, i, None], torch.zeros_like(v), v)
            v, r = model.core_step(v, x[:, i])
            rs.append(r)
    r = torch.stack(rs, 1)
    dec = r[:, clocks.decision_core_steps(n_dec)]
    with torch.no_grad():
        dist = model.actor.distribution(dec)
        a = dist.sample()
        buf = RolloutBuffer(
            t4t5=x, rel_yaw=torch.rand(n_env, t, generator=g), episode_start=start, v0=v0, actions=a,
            logp=dist.log_prob(a).sum(-1), values=model.critic(dec), rewards=torch.randn(n_env, n_dec, generator=g),
            dones=torch.zeros(n_env, n_dec, dtype=torch.bool), bc_mean=dist.mean, last_value=torch.zeros(n_env),
        )
    return buf, r


def test_rerun_matches_the_rollout(cfg, model):
    buf, r = fake_rollout(model, cfg)
    with torch.no_grad():
        r2, dec = rerun(model, buf, torch.arange(3), 16, clocks_from_cfg(cfg))
    torch.testing.assert_close(r2, r, atol=1e-5, rtol=1e-5)
    logp = model.actor.distribution(dec).log_prob(buf.actions).sum(-1)
    torch.testing.assert_close(logp, buf.logp, atol=1e-4, rtol=1e-4)


def test_warmup_trains_only_the_critic(cfg, model):
    buf, _ = fake_rollout(model, cfg)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    actor = torch.optim.Adam([q for n, q in model.named_parameters() if not n.startswith("critic.")], lr=1e-2)
    critic = torch.optim.Adam(model.critic.parameters(), lr=1e-2)
    ppo_update(model, buf, actor, critic, cfg, update=0)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all(k.startswith("critic.") for k in changed)

    stats = ppo_update(model, buf, actor, critic, cfg, update=int(cfg.ppo.critic_warmup_iters))
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert "core.W" in changed and "actor.linear.weight" in changed
    assert np.isfinite(stats["pg_loss"]) and stats["kl_bc"] >= 0


def test_recordings_stop_at_the_budget(tmp_path):
    from flybywire.train.watch import room_to_record

    assert room_to_record(tmp_path, budget_gb=1e-6, min_free_gb=0)
    (tmp_path / "update_0000.mp4").write_bytes(b"x" * 2000)
    assert not room_to_record(tmp_path, budget_gb=1e-6, min_free_gb=0)
    assert not room_to_record(tmp_path, budget_gb=1.0, min_free_gb=1e9)
