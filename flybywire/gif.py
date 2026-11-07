"""GIF export for the README media."""

from __future__ import annotations

import subprocess

import numpy as np


def save_gif(frames, path, fps: float) -> None:
    """Write same-sized RGB frames as a looping GIF at full resolution.

    imageio's Pillow writer gives every frame its own undithered 256-colour palette, which bands
    gradients and makes colours flicker; ffmpeg builds one palette for the clip and dithers to it.
    """
    from imageio_ffmpeg import get_ffmpeg_exe

    h, w = frames[0].shape[:2]
    graph = "split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    cmd = [
        get_ffmpeg_exe(), "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
        "-vf", graph, "-loop", "0", str(path),
    ]  # fmt: skip
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f[..., :3], dtype=np.uint8).tobytes())
        proc.stdin.close()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)
