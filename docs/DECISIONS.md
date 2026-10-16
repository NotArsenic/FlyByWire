# Decisions log

Append-only. Each entry: date, decision, rationale. Deviations from the setup spec are marked
**[deviation]**.

---

## 2026-10-16 — D1. A single recurrent population, not stacked residual blocks

The central complex core is one population of N rate neurons whose only neuron-to-neuron path is
`r @ (W * M)`. No residual stream, no LayerNorm, no attention, no dense layer over the neuron axis.

*Rationale.* A residual stream would carry EPG activity around the mask, so silencing EPG would not
actually silence its downstream influence and lesion experiments would be uninterpretable.
LayerNorm across neurons is itself an all-to-all interaction that bypasses the connectome. With a
single masked population, "which neurons can influence which" is exactly the FlyWire mask.

## 2026-10-16 — D2. Directed, degree-preserving rewire; `networkx.double_edge_swap` rejected

The null model uses repeated directed two-edge swaps (`a→b, c→d` ⇒ `a→d, c→b`, four distinct
endpoints, reject if a new edge exists or is a self-loop), 10 × |E| successful swaps, seeded.

*Rationale.* This preserves every node's in-degree and out-degree exactly, so the null differs from
the real connectome only in *which* partners are connected. `networkx.double_edge_swap` works on
undirected graphs only and would destroy the in/out-degree split that defines e.g. ER as pure
input and PFL3 as mostly output neurons. An Erdős–Rényi (`er`) mask with the same N and edge count
is kept as a weaker second null.

## 2026-10-16 — D3. Body-frame velocity + yaw-rate action space

Policy outputs `[-1, 1]^4` → `[v_forward, v_lateral, v_vertical, yaw_rate]` (body frame), tracked by
gym-pybullet-drones' PID controller (target yaw integrated from `yaw_rate`, body velocity rotated
into a world-frame target velocity).

*Rationale.* Raw thrust or body-rate control would require writing an inner-loop attitude
controller and would make both the scripted expert and BC fragile. Yaw rate remains the steering
variable that PFL3 controls, which is the biological claim of interest.

## 2026-10-16 — D4. Heading relative to the episode start; whole-episode BC

The heading target is `(sin, cos)(wrap(yaw_t − yaw_0))`. BC sequences always start at step 0 of an
episode; truncated BPTT carries state across 128-step windows and detaches at window boundaries.

*Rationale.* The network has no compass input; an absolute heading is not observable from optic
flow alone. Heading relative to start *is* observable by integrating yaw rate, but only if the core
integrates from t = 0 — random mid-episode chunks would give the core an unknown offset.

## 2026-10-16 — D5. LPTC `gains` mode is the default

LPTC weights are fixed ego-motion templates (Longuet-Higgins & Prazdny flow fields projected onto
calibrated T4/T5 preferred directions); only a per-(channel, subtype) gain (8 × 8 = 64 params) is
trained by default. `frozen` (0 params) and `full` (~46k params) are available.

*Rationale.* Keeps the connectome-masked core as the dominant trainable component, so differences
between mask conditions are not absorbed by a large learnable visual front end.
`scripts/count_params.py` reports the ratio.

## 2026-10-16 — D6. fp32 only

No AMP, no autocast, no bf16, no `torch.compile` (for now). A test asserts every parameter and buffer
is fp32. (The on-disk T4/T5 cache and stored hexals are fp16 for size; they are upcast to fp32 on
load before entering the model.)

*Rationale.* The ring-attractor dynamics are marginally stable; low-precision rounding produces slow
drift of the bump that looks biological but is a numerical artefact.

## 2026-10-16 — D7. The controlled comparison runs in the BC regime only

Real vs. rewired (and optionally ER) masks are compared with 3 seeds each under behavior cloning.
PPO fine-tuning is run on the real connectome for the demo.

*Rationale.* BC is a supervised problem with low variance across seeds, so mask effects are
measurable with 3 seeds; PPO variance would swamp them at this budget. One cache serves every
condition and seed.

---

## Installation and environment

## 2026-10-16 — D8. Python 3.12 instead of 3.10 **[deviation]**

The spec suggests Python 3.10 "to match gym-pybullet-drones' documented setup". The current
`main` branch of gym-pybullet-drones (`pyproject.toml`, version 2.2.0) declares `python = "^3.12"`,
and pip enforces `Requires-Python` even with `--no-deps`. flyvis 1.2.0 declares
`>=3.9,<3.13`. The only Python satisfying both is **3.12**, so `environment.yml` pins
`python=3.12`.

## 2026-10-16 — D9. PyTorch from the CUDA 13.0 index (cu130)

PyTorch's release matrix lists 2.14 with stable CUDA 12.6 / 13.0 / 13.2; the cu128 index tops out at
torch 2.11. gym-pybullet-drones declares `torch = "^2.13"`. The spec allows "a newer CUDA index if
one is the current stable", so `setup_env.sh` uses `https://download.pytorch.org/whl/cu130`
(override with `TORCH_INDEX=...`). The RTX 5060 Ti (sm_120) needs CUDA ≥ 12.8; cu130 satisfies this.
`check_env.py` verifies `sm_120` is in `torch.cuda.get_arch_list()`.
