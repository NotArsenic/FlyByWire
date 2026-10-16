#!/usr/bin/env python3
"""Preflight / environment check for Fly-By-Wire.

Safe to rerun at any time. Prints a PASS / WARN / FAIL / SKIP table.

System-level checks (stdlib only, run anywhere):
  WSL detection, repo location, memory, disk, CPU topology, GPU (nvidia-smi).
Environment checks (run once the conda env exists; SKIP if packages are missing):
  PyTorch + CUDA arch support + GPU matmul, flyvis + pretrained models,
  PyBullet import + EGL vs TinyRenderer timing, gymnasium version.

Usage:
  python scripts/check_env.py            # all checks
  python scripts/check_env.py --system   # system-level checks only
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROWS: list[tuple[str, str, str]] = []
NOTES: list[str] = []

WSLCONFIG_SNIPPET = """\
Add this to %UserProfile%\\.wslconfig on Windows (create the file if needed):

    [wsl2]
    memory=24GB
    swap=8GB

then run in PowerShell:  wsl --shutdown"""


def row(name: str, status: str, detail: str) -> None:
    ROWS.append((name, status, detail))


def _run(cmd: list[str], timeout: float = 20.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True)
        return out.stdout
    except Exception:
        return None


# --------------------------------------------------------------------------- system


def check_wsl() -> None:
    try:
        version = Path("/proc/version").read_text()
    except OSError:
        row("WSL", "SKIP", "/proc/version not readable (not Linux?)")
        return
    is_wsl = "microsoft" in version.lower() or "wsl" in version.lower()
    if not is_wsl:
        row("WSL", "PASS", "native Linux (not WSL)")
        return
    row("WSL", "PASS", "WSL detected")
    if str(REPO).startswith("/mnt/"):
        row("WSL repo location", "FAIL", f"repo is under /mnt/ ({REPO}); move it to the Linux filesystem (~/...)")
    else:
        row("WSL repo location", "PASS", str(REPO))
    NOTES.append("WSL: the WSL virtual disk (ext4.vhdx) lives on C: by default; data written here consumes C: space.")


def check_memory() -> None:
    try:
        meminfo = Path("/proc/meminfo").read_text()
        kb = int(re.search(r"MemTotal:\s+(\d+)", meminfo).group(1))
    except Exception:
        row("Memory", "SKIP", "could not read /proc/meminfo")
        return
    gb = kb / 1024**2
    is_wsl = "microsoft" in Path("/proc/version").read_text().lower()
    if is_wsl and gb < 23:
        row("Memory", "WARN", f"MemTotal {gb:.1f} GB < 23 GB under WSL")
        NOTES.append(WSLCONFIG_SNIPPET)
    elif gb < 23:
        row("Memory", "WARN", f"MemTotal {gb:.1f} GB < 23 GB (PPO with 12 envs may be tight)")
    else:
        row("Memory", "PASS", f"MemTotal {gb:.1f} GB")


def check_disk() -> None:
    usage = shutil.disk_usage(REPO)
    free_gb = usage.free / 1e9
    status = "PASS" if free_gb >= 50 else "WARN"
    row("Disk free", status, f"{free_gb:.0f} GB free at {REPO} (need >= 50 GB)")


def check_cpu() -> None:
    n = os.cpu_count()
    out = _run(["lscpu", "-e=CPU,CORE,MAXMHZ"])
    if out is None:
        row("CPU", "PASS", f"{n} logical CPUs (lscpu unavailable)")
        return
    lines = [ln.split() for ln in out.strip().splitlines()[1:]]
    maxmhz: dict[float, set[str]] = {}
    cpus_by_mhz: dict[float, int] = {}
    for parts in lines:
        if len(parts) < 3:
            continue
        try:
            mhz = round(float(parts[2]))
        except ValueError:
            continue
        maxmhz.setdefault(mhz, set()).add(parts[1])
        cpus_by_mhz[mhz] = cpus_by_mhz.get(mhz, 0) + 1
    summary = ", ".join(
        f"{cpus_by_mhz[m]} threads/{len(maxmhz[m])} cores @ {m} MHz" for m in sorted(maxmhz, reverse=True)
    )
    hint = ""
    if len(maxmhz) >= 2:
        # Hybrid Intel parts: P-cores have SMT (2 threads/core) and the higher max clock.
        hi = [m for m in maxmhz if cpus_by_mhz[m] == 2 * len(maxmhz[m])]
        lo = [m for m in maxmhz if cpus_by_mhz[m] == len(maxmhz[m])]
        if hi and lo:
            p_threads = sum(cpus_by_mhz[m] for m in hi)
            e_threads = sum(cpus_by_mhz[m] for m in lo)
            hint = f"; hint: ~{p_threads} P-core threads (SMT, >= {min(hi)} MHz), {e_threads} E-cores"
    row("CPU", "PASS", f"{n} logical CPUs: {summary}{hint}")


def check_gpu() -> None:
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if out is None:
        row("GPU (nvidia-smi)", "FAIL", "nvidia-smi not found or failed")
        return
    for i, line in enumerate(out.strip().splitlines()):
        row(f"GPU {i}", "PASS", line.strip())


# --------------------------------------------------------------------------- env


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        row("PyTorch", "SKIP", "torch not importable (environment not created / not activated?)")
        return
    row("PyTorch", "PASS", f"torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        row("CUDA available", "FAIL", "torch.cuda.is_available() is False")
        return
    cap = torch.cuda.get_device_capability(0)
    arch = f"sm_{cap[0]}{cap[1]}"
    archs = torch.cuda.get_arch_list()
    row("CUDA available", "PASS", f"{torch.cuda.get_device_name(0)}, capability {cap[0]}.{cap[1]}")
    if arch in archs:
        row("CUDA arch in build", "PASS", f"{arch} in {archs}")
    else:
        row("CUDA arch in build", "FAIL", f"{arch} NOT in {archs}; install a wheel built for this GPU (cu128+)")
    try:
        a = torch.randn(512, 512, device="cuda")
        b = (a @ a).sum().item()
        torch.cuda.synchronize()
        row("GPU matmul", "PASS", f"512x512 matmul ok (checksum {b:.3g})")
    except Exception as e:  # noqa: BLE001
        row("GPU matmul", "FAIL", f"{type(e).__name__}: {e}")


def check_flyvis() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    os.environ.setdefault("FLYVIS_ROOT_DIR", str(REPO / "data" / "flyvis"))
    try:
        import flyvis  # noqa: F401
        from importlib.metadata import version

        row("flyvis import", "PASS", f"flyvis {version('flyvis')}")
    except Exception as e:  # noqa: BLE001
        row("flyvis import", "SKIP" if isinstance(e, ImportError) else "FAIL", f"{type(e).__name__}: {e}")
        return
    root = Path(os.environ["FLYVIS_ROOT_DIR"])
    if not root.is_absolute():
        root = (REPO / root).resolve()
    model_dir = root / "results" / "flow" / "0000" / "000"
    if model_dir.exists() and any(model_dir.rglob("*")):
        row("flyvis pretrained", "PASS", f"found {model_dir.relative_to(REPO) if model_dir.is_relative_to(REPO) else model_dir}")
    else:
        row("flyvis pretrained", "FAIL", f"missing {model_dir}; run `make env` step 7 (flyvis download-pretrained)")


def _time_render(p, renderer: int, n: int = 30) -> float:
    view = p.computeViewMatrix([0, 0, 1], [1, 0, 1], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(110, 1.0, 0.05, 50)
    p.getCameraImage(160, 160, view, proj, renderer=renderer)  # warmup
    t0 = time.perf_counter()
    for _ in range(n):
        p.getCameraImage(160, 160, view, proj, renderer=renderer)
    return (time.perf_counter() - t0) / n * 1e3


def check_pybullet() -> None:
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as e:
        row("PyBullet", "SKIP", f"{e}")
        return
    row("PyBullet", "PASS", f"pybullet API {p.getAPIVersion()}")
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.loadURDF("r2d2.urdf", [2, 0, 0.5])
        tiny_ms = _time_render(p, p.ER_TINY_RENDERER)
        egl_ms = None
        try:
            import importlib.util

            egl = importlib.util.find_spec("eglRenderer")
            plugin = p.loadPlugin(egl.origin, "_eglRendererPlugin") if egl else -1
            if plugin >= 0:
                egl_ms = _time_render(p, p.ER_BULLET_HARDWARE_OPENGL)
                p.unloadPlugin(plugin)
        except Exception:  # noqa: BLE001
            egl_ms = None
        if egl_ms is None:
            row("PyBullet EGL", "WARN", f"EGL plugin failed to load; TinyRenderer {tiny_ms:.2f} ms/frame @160x160")
        else:
            row(
                "PyBullet EGL",
                "PASS",
                f"EGL {egl_ms:.2f} ms/frame vs TinyRenderer {tiny_ms:.2f} ms/frame @160x160",
            )
    finally:
        p.disconnect(cid)


def check_gymnasium() -> None:
    try:
        import gymnasium
    except ImportError as e:
        row("gymnasium", "SKIP", f"{e}")
        return
    row("gymnasium", "PASS", f"gymnasium {gymnasium.__version__}")
    try:
        import gym_pybullet_drones  # noqa: F401
        from importlib.metadata import version

        row("gym-pybullet-drones", "PASS", f"gym-pybullet-drones {version('gym-pybullet-drones')}")
    except Exception as e:  # noqa: BLE001
        row("gym-pybullet-drones", "SKIP", f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- main


def print_table() -> None:
    w0 = max(len(r[0]) for r in ROWS)
    colors = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m", "SKIP": "\033[90m"}
    reset = "\033[0m" if sys.stdout.isatty() else ""
    print(f"\n{'check'.ljust(w0)}  status  detail")
    print("-" * (w0 + 2) + "  ------  " + "-" * 50)
    for name, status, detail in ROWS:
        c = colors.get(status, "") if sys.stdout.isatty() else ""
        print(f"{name.ljust(w0)}  {c}{status.ljust(6)}{reset}  {detail}")
    if NOTES:
        print("\nNotes:")
        for n in NOTES:
            print("  " + n.replace("\n", "\n  "))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", action="store_true", help="system-level checks only")
    args = ap.parse_args()

    row("Python", "PASS", f"{sys.version.split()[0]} at {sys.executable}")
    check_wsl()
    check_memory()
    check_disk()
    check_cpu()
    check_gpu()
    if not args.system:
        check_torch()
        check_flyvis()
        check_pybullet()
        check_gymnasium()
    print_table()
    return 1 if any(r[1] == "FAIL" for r in ROWS) else 0


if __name__ == "__main__":
    sys.exit(main())
