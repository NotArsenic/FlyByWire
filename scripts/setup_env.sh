#!/usr/bin/env bash
# Create the `flybywire` conda env and install everything in a fixed order.
# Safe to rerun: an existing env is reused. Blackwell GPUs need a torch built for CUDA >= 12.8.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
ENV_NAME="${ENV_NAME:-flybywire}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
GPD_URL="${GPD_URL:-git+https://github.com/utiasDSL/gym-pybullet-drones}"

log() { printf '\n\033[1;34m[setup]\033[0m %s\n' "$*"; }

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

log "1/8 conda env '$ENV_NAME'"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env exists; reusing"
else
  conda env create -f environment.yml -n "$ENV_NAME"
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip

log "2/8 PyTorch from $TORCH_INDEX"
python -m pip install torch torchvision --index-url "$TORCH_INDEX"

torch_ok() {
  python - <<'PY'
import sys, torch
cuda = torch.version.cuda
print(f"torch {torch.__version__} cuda {cuda}")
ok = cuda is not None and tuple(int(x) for x in cuda.split(".")[:2]) >= (12, 8)
sys.exit(0 if ok else 1)
PY
}
torch_ok
TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"

log "3/8 flyvis"
python -m pip install flyvis
if ! torch_ok || [ "$(python -c 'import torch; print(torch.__version__)')" != "$TORCH_VER" ]; then
  echo "!! flyvis dependencies replaced torch; reinstalling from $TORCH_INDEX"
  python -m pip install --force-reinstall --no-deps torch torchvision --index-url "$TORCH_INDEX"
  torch_ok
fi

log "4/8 gym-pybullet-drones"
# Dry-run first: if installing with dependencies would replace any package that is
# already installed (torch, numpy, numba, ...), install with --no-deps instead.
REPORT="$(mktemp)"
python -m pip install --dry-run --quiet --report "$REPORT" "$GPD_URL"
CHANGES="$(python scripts/_pip_report_changes.py "$REPORT")"
rm -f "$REPORT"
if [ -z "$CHANGES" ]; then
  python -m pip install "$GPD_URL"
  echo "gym-pybullet-drones installed with its dependencies"
else
  echo "!! installing gym-pybullet-drones with deps would replace:"
  echo "$CHANGES" | sed 's/^/     /'
  echo "!! using --no-deps and installing its runtime deps explicitly"
  python -m pip install --no-deps "$GPD_URL"
  # runtime deps of gym_pybullet_drones (stable-baselines3 is only used by its examples)
  python -m pip install pillow transforms3d control
fi
torch_ok

log "5/8 requirements.txt"
python -m pip install -r requirements.txt

log "6/8 editable install"
python -m pip install -e .

log "7/8 flyvis pretrained models"
[ -f .env ] || cp .env.example .env
mkdir -p data/flyvis
export FLYVIS_ROOT_DIR="$REPO/data/flyvis"
flyvis download-pretrained --skip_large_files
du -sh data/flyvis

log "8/8 lock file"
torch_ok
python -m pip freeze --exclude-editable > requirements.lock.txt
echo "done. activate with: conda activate $ENV_NAME"
