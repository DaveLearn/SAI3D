#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MSDA_DIR="$ROOT_DIR/Semantic-SAM/semantic_sam/body/encoder/ops"
MSDA_SO="$MSDA_DIR/MultiScaleDeformableAttention*.so"

# Decide the CUDA arch list explicitly rather than inheriting one.
#
# conda's cuda-nvcc activation exports TORCH_CUDA_ARCH_LIST only "if unset", so
# whatever the launching shell had wins. Launching this from an activated root
# project environment leaks its arch list into this project's nvcc 12.6, and an
# arch such as 10.1/12.0 fails with "Unsupported gpu architecture". Clearing it
# lets torch auto-detect the local GPU, which this toolchain can always build.
#
#   pixi run build_deps                                # auto-detect
#   SAI3D_CUDA_ARCH_LIST="8.0;8.6" pixi run build_deps # explicit
if [ -n "${SAI3D_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST="${SAI3D_CUDA_ARCH_LIST}"
  printf '[build_deps] building for arch list: %s\n' "${TORCH_CUDA_ARCH_LIST}"
else
  unset TORCH_CUDA_ARCH_LIST
  printf '[build_deps] building for auto-detected GPU arch\n'
fi

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
  # NOTE: do not use the vendored make.sh -- it runs `setup.py install --user`,
  # which drops the egg in ~/.local/lib/python3.11/site-packages where it is
  # picked up by every other python3.11 environment on the machine and is built
  # against this env's torch ABI. Install into the pixi env instead.
  python -m pip install --no-build-isolation --no-deps --no-cache-dir "$MSDA_DIR"
  echo "[build_deps] Build complete."
fi
