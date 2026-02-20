#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSDA_DIR="$ROOT_DIR/Semantic-SAM/semantic_sam/body/encoder/ops"
MSDA_SO="$MSDA_DIR/MultiScaleDeformableAttention*.so"

# Ensure CUDA toolkit from pixi/conda is discoverable
if [ -n "${CONDA_PREFIX:-}" ]; then
  if [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
  elif [ -f "$CONDA_PREFIX/include/cuda_runtime.h" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
  elif [ -f "$CONDA_PREFIX/targets/x86_64-linux/include/cuda_runtime.h" ]; then
    export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
  fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
  fi
fi

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
  NVCC_PATH="$(command -v nvcc)"
  NVCC_BIN_DIR="$(dirname "$NVCC_PATH")"
  NVCC_PREFIX="$(cd "$NVCC_BIN_DIR/.." && pwd)"
  if [ -f "$NVCC_PREFIX/include/cuda_runtime.h" ]; then
    export CUDA_HOME="$NVCC_PREFIX"
  elif [ -f "$NVCC_PREFIX/targets/x86_64-linux/include/cuda_runtime.h" ]; then
    export CUDA_HOME="$NVCC_PREFIX/targets/x86_64-linux"
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  if [ -d "$CUDA_HOME/targets/x86_64-linux/include" ]; then
    export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
  elif [ -d "$CUDA_HOME/include" ]; then
    export CPATH="$CUDA_HOME/include:${CPATH:-}"
  fi
  if [ -d "$CUDA_HOME/targets/x86_64-linux/lib" ]; then
    export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
  fi
fi

echo "[build_deps] Checking MultiScaleDeformableAttention CUDA op..."
if python - <<'PY' >/dev/null 2>&1
import importlib.util
spec = importlib.util.find_spec("MultiScaleDeformableAttention")
raise SystemExit(0 if spec is not None else 1)
PY
then
  echo "[build_deps] MultiScaleDeformableAttention already installed."
elif ls $MSDA_SO >/dev/null 2>&1; then
  echo "[build_deps] MultiScaleDeformableAttention already built."
else
  if [ ! -d "$MSDA_DIR" ]; then
    echo "[build_deps] ERROR: $MSDA_DIR not found."
    exit 1
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[build_deps] Skipping MultiScaleDeformableAttention build: nvcc not found."
    echo "[build_deps] Install CUDA toolkit in the pixi env or set CUDA_HOME to enable this build."
    exit 0
  fi
  echo "[build_deps] Building MultiScaleDeformableAttention..."
  (cd "$MSDA_DIR" && bash make.sh)
  echo "[build_deps] Build complete."
fi
