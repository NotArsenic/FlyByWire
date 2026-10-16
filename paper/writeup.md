# Fly-By-Wire: steering a simulated quadcopter with a connectome-masked central complex

> Skeleton. No results yet; nothing here is a claim.

## Abstract

TBD.

## Introduction

TBD.

## Related Work

TBD. (Connectome-constrained vision models; central-complex heading circuits; LPTC self-motion
estimation; bio-inspired drone control. See `references.md`.)

## Methods

### Environment

PyBullet quadcopter (gym-pybullet-drones, CF2X) in a 30 m × 3 m × 2.5 m obstacle corridor; forward
FPV camera, 110° FOV, 160×160 grayscale. Body-frame velocity + yaw-rate actions tracked by the
library PID controller.

### Vision

Frozen pretrained flyvis model `flow/0000/000` (Lappalainen et al., 2024) driven through a BoxEye
(extent 15, kernel 13) at 100 Hz; T4a–d and T5a–d outputs are read out.

### LPTC

Fixed ego-motion flow templates (Longuet-Higgins & Prazdny, 1980) projected onto calibrated T4/T5
preferred directions; per-(channel, subtype) learnable gains.

### CX core

Single recurrent rate population masked by the FlyWire central-complex connectome
(ER, EPG, PEN, PEG, Δ7, PFL3). `v_{t+1} = (1 - α) v_t + α (φ(v_t) @ (W ⊙ M) + u_t + b)`.

### Training

Behavior cloning from a scripted expert on whole episodes (truncated BPTT, auxiliary heading
loss), followed by PPO fine-tuning with a KL anchor to the BC policy.

### Null models

Directed degree-preserving rewiring (two-edge swaps) and an Erdős–Rényi control with matched N and
edge count.

### Lesions

Inference-time silencing of EPG, PEN or Δ7 by zeroing their rates at every step.

## Results

TBD.

- **Fig 1:** Real vs. rewired adjacency matrices. *(placeholder)*
- **Fig 2:** Ring bump against true heading over time. *(placeholder)*
- **Fig 3:** Heading decode per condition, with per-seed points. *(placeholder)*
- **Table 1:** Lesion effects. *(placeholder)*
- **Fig 4:** BC→PPO success rate. *(placeholder)*

## Limitations

TBD.

## Conclusion

TBD.
