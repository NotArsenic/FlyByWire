"""Do EPG neurons form a heading ring that matches their anatomical angle around the EB?

Two angles per EPG neuron are compared: the anatomical one (centroid of its EB arbor, from
``scripts/epg_anatomy.py``) and the preferred heading found in the trained network, either from
the rates (least squares of rate on ``[1, sin h, cos h]``) or from the heading-decoder weights.

The anatomical frame has an arbitrary zero and handedness, so scores are invariant to rotation
and reflection and use label-shuffle permutation tests. Tuning fits are kept as per-episode
moments so a bootstrap over episodes is a cheap resampling of sums.
"""

from __future__ import annotations

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def circ_mean(a, axis=None):
    return np.angle(np.exp(1j * np.asarray(a)).mean(axis=axis))


def circ_corr(a, b) -> float:
    """Circular-circular correlation (Jammalamadaka and SenGupta)."""
    sa = np.sin(np.asarray(a) - circ_mean(a))
    sb = np.sin(np.asarray(b) - circ_mean(b))
    return float((sa * sb).sum() / np.sqrt((sa**2).sum() * (sb**2).sum()))


def alignment(phi, theta) -> tuple[float, int, float]:
    """Best match of phi to theta up to rotation and reflection.

    Returns (score, sign, offset): score = max over s in {+1, -1} of |mean exp(i (phi - s theta))|,
    and phi is roughly ``sign * theta + offset``.
    """
    best = (-1.0, 1, 0.0)
    for s in (1, -1):
        z = np.exp(1j * (np.asarray(phi) - s * np.asarray(theta))).mean()
        if abs(z) > best[0]:
            best = (float(abs(z)), s, float(np.angle(z)))
    return best


def alignment_score(phi, theta) -> float:
    return alignment(phi, theta)[0]


def abs_circ_corr(a, b) -> float:
    return abs(circ_corr(a, b))


def harmonic_r2(y, theta) -> float:
    """R^2 of y on [1, cos theta, sin theta].

    Invariant to rotation and reflection of theta, so it asks "does y vary as a sinusoid of the
    anatomical angle" without needing the zero or the handedness.
    """
    y = np.asarray(y, dtype=float)
    x = np.stack([np.ones_like(y), np.cos(theta), np.sin(theta)], axis=1)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    sst = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - x @ beta) ** 2).sum() / sst) if sst > 0 else 0.0


def permutation_test(stat, phi, theta, n_perm: int = 10000, seed: int = 0) -> dict:
    """Shuffle neuron labels of theta. p is one-sided (null >= observed), with the +1 correction."""
    rng = np.random.default_rng(seed)
    obs = stat(phi, theta)
    null = np.array([stat(phi, rng.permutation(theta)) for _ in range(n_perm)])
    return {"value": float(obs), "p": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "null_mean": float(null.mean()), "null_95": float(np.quantile(null, 0.95))}


def episode_moments(rate, heading) -> dict:
    """Sums for the least-squares fit rate ~ 1 + sin h + cos h on one episode.

    rate: (T, n) rates, heading: (T,) radians. Float64 because sin and cos are nearly collinear
    with the intercept when headings are small.
    """
    rate = np.asarray(rate, dtype=np.float64)
    h = np.asarray(heading, dtype=np.float64)
    x = np.stack([np.ones_like(h), np.sin(h), np.cos(h)], axis=1)
    return {"xtx": x.T @ x, "xty": x.T @ rate, "yty": (rate**2).sum(0), "n": float(len(h))}


def stack_moments(moments: list[dict]) -> dict:
    return {k: np.stack([m[k] for m in moments]) for k in moments[0]}


def fit_tuning(stacked: dict, weights=None) -> dict:
    """Per-neuron fit from stacked per-episode moments.

    ``weights`` (episodes,) are bootstrap counts; default is every episode once. phi is the
    preferred heading, amplitude / mean the relative tuning strength, slope the coefficient on
    sin h in units of the neuron's rate sd (the part of the tuning small headings can pin down).
    """
    w = np.ones(len(stacked["n"])) if weights is None else np.asarray(weights, dtype=np.float64)
    xtx = np.einsum("e,eij->ij", w, stacked["xtx"])
    xty = np.einsum("e,ein->in", w, stacked["xty"])
    yty = w @ stacked["yty"]
    n = w @ stacked["n"]
    beta = np.linalg.solve(xtx, xty)
    sst = yty - xty[0] ** 2 / n
    sse = yty - (beta * xty).sum(0)
    mean = xty[0] / n
    amp = np.hypot(beta[1], beta[2])
    sd = np.sqrt(np.maximum(sst, 1e-30) / n)
    return {
        "phi": np.arctan2(beta[1], beta[2]),
        "amplitude": amp,
        "mean": mean,
        "strength": amp / np.maximum(mean, 1e-12),
        "r2": 1 - sse / np.maximum(sst, 1e-30),
        "slope": beta[1] / sd,
    }


def val_split(n: int, seed: int, fraction: float = 0.1) -> np.ndarray:
    """Validation episode indices, chosen the way ``train_bc`` chooses them."""
    n_val = int(round(fraction * n)) if n >= 10 else 0
    return np.random.default_rng(seed).permutation(n)[:n_val]


def decoder_angle(weight):
    """Angle of each EPG neuron's decoder weights; weight rows are (sin, cos), shape (2, n)."""
    w = np.asarray(weight)
    return np.arctan2(w[0], w[1])


def bootstrap_tuning(stacked: dict, n_boot: int = 500, seed: int = 0) -> dict[str, np.ndarray]:
    """Resample episodes with replacement; returns each fit field stacked over draws."""
    rng = np.random.default_rng(seed)
    e = len(stacked["n"])
    draws = [fit_tuning(stacked, np.bincount(rng.integers(0, e, e), minlength=e)) for _ in range(n_boot)]
    return {k: np.stack([d[k] for d in draws]) for k in draws[0]}


def cos_over_sin_se(stacked: dict) -> float:
    """Standard error of the cos h coefficient relative to the sin h one (same for every neuron).

    Large when headings are small: cos h varies little and is nearly collinear with the intercept.
    """
    inv = np.linalg.inv(stacked["xtx"].sum(0))
    return float(np.sqrt(inv[2, 2] / inv[1, 1]))


def expected_alignment(theta, sd, n_draws: int = 1000, seed: int = 0) -> np.ndarray:
    """Alignment scores if phi were exactly theta plus Gaussian error with per-neuron sd (rad).

    Shows what a perfect ring would score given how noisy the phi estimates are.
    """
    rng = np.random.default_rng(seed)
    return np.array([alignment_score(theta + rng.normal(0, sd), theta) for _ in range(n_draws)])


def phi_spread(phi_hat, phi_boot) -> np.ndarray:
    """Half-width (rad) of the central 95% of bootstrap phi around the full-sample phi, per neuron."""
    return np.quantile(np.abs(wrap(phi_boot - phi_hat)), 0.95, axis=0)


def ring_frame(points, center):
    """Frame of the ring plane fitted by PCA. Returns (axes (3, 3), sd (2,)).

    axes rows are lateral (toward +x), dorsal (toward -y, projected into the plane) and the
    normal (smallest-variance PCA axis). The in-plane axes come from the anatomy rather than the
    PCA because a nearly round ring has no stable principal axes in its plane. sd is the spread
    of the points along lateral and dorsal.
    """
    d = np.asarray(points) - center
    normal = np.linalg.svd(d, full_matrices=False)[2][2]
    dorsal = np.array([0.0, -1.0, 0.0])
    dorsal = dorsal - (dorsal @ normal) * normal
    dorsal /= np.linalg.norm(dorsal)
    lateral = np.cross(dorsal, normal)
    if lateral[0] < 0:
        lateral = -lateral
    axes = np.stack([lateral, dorsal, normal])
    return axes, (d @ axes[:2].T).std(0)


def ring_angles(points, center, axes, sd) -> np.ndarray:
    """Angle of each point around the ring: 0 points dorsal, positive turns toward lateral (+x).

    In-plane coordinates are divided by sd, so an elliptical ring maps to a circle.
    """
    d = np.asarray(points) - center
    return np.arctan2((d @ axes[0]) / sd[0], (d @ axes[1]) / sd[1])
