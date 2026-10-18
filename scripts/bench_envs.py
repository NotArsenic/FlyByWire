#!/usr/bin/env python
"""Benchmark corridor-env throughput. Not implemented yet.

Should measure, for n_envs in {1, 4, 8, 12}:
  * decisions/s of ``AsyncVectorEnv(shared_memory=True)`` with random actions (frames only),
  * renderer in use (EGL vs TinyRenderer) and its ms/frame,
  * end-to-end decisions/s including batched BoxEye + flyvis + core on the GPU.
Writes runs/bench_envs/report.json.

Usage: python scripts/bench_envs.py [--n-envs 1 4 8 12] [key=value ...]
"""

from __future__ import annotations


def main(argv=None) -> int:
    raise NotImplementedError("bench_envs is not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
