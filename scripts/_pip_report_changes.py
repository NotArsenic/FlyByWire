"""Helper for setup_env.sh: list already-installed packages a `pip install --report` would replace."""

import json
import sys
from importlib.metadata import PackageNotFoundError, version


def main(path: str) -> None:
    with open(path) as f:
        report = json.load(f)
    for item in report.get("install", []):
        name = item["metadata"]["name"]
        new = item["metadata"]["version"]
        if name.lower().replace("_", "-") == "gym-pybullet-drones":
            continue
        try:
            cur = version(name)
        except PackageNotFoundError:
            continue
        if cur != new:
            print(f"{name} {cur} -> {new}")


if __name__ == "__main__":
    main(sys.argv[1])
