#!/usr/bin/env bash
# One Linux entrypoint: environment -> tests -> pins/data -> 4 GPU inference -> CPU statistics -> bounded exports.
set -euo pipefail
set +x
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
RUNTIME="${LLADA_RUNTIME_DIR:-$ROOT/.runtime}"
mkdir -p "$RUNTIME"
# Do not let two launchers change the same environment concurrently.
exec 9>"$RUNTIME/environment.lock"
flock 9
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "This launcher requires Linux x86_64. CPU source tests also run on Windows." >&2; exit 2
fi
CONDA_BIN="$RUNTIME/miniforge/bin/conda"
if [[ ! -x "$CONDA_BIN" ]]; then
  RELEASE="25.3.1-0"
  NAME="Miniforge3-${RELEASE}-Linux-x86_64.sh"
  URL="https://github.com/conda-forge/miniforge/releases/download/${RELEASE}/${NAME}"
  curl --fail --location --retry 5 "$URL" -o "$RUNTIME/$NAME"
  printf "376b160ed8130820db0ab0f3826ac1fc85923647f75c1b8231166e3d559ab768  %s\n" "$RUNTIME/$NAME" | sha256sum --check -
  bash "$RUNTIME/$NAME" -b -p "$RUNTIME/miniforge"
fi
ENV="$RUNTIME/env"
if [[ ! -x "$ENV/bin/python" ]]; then
  "$CONDA_BIN" create --yes --prefix "$ENV" --override-channels -c conda-forge python=3.11.11 pip=25.0.1
fi
PYTHON="$ENV/bin/python"
EXPECTED="$( (cat requirements.txt; printf '\ntorch==2.5.1+cu124\n') | sha256sum | cut -d' ' -f1)"
if [[ ! -f "$RUNTIME/INSTALLED" || "$(cat "$RUNTIME/INSTALLED")" != "$EXPECTED" ]]; then
  "$PYTHON" -m pip install --disable-pip-version-check torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
  "$PYTHON" -m pip install --disable-pip-version-check -r requirements.txt
  "$PYTHON" -m pip check
  printf '%s' "$EXPECTED" > "$RUNTIME/INSTALLED"
fi
# Repair ordinary package drift instead of trusting the installation sentinel.
if ! "$PYTHON" -m llada_study.environment_check; then
  "$CONDA_BIN" install --yes --prefix "$ENV" --override-channels -c conda-forge python=3.11.11 pip=25.0.1
  "$PYTHON" -m pip install --disable-pip-version-check torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
  "$PYTHON" -m pip install --disable-pip-version-check -r requirements.txt
fi
"$PYTHON" -m llada_study.environment_check
"$PYTHON" -m pip check
flock -u 9
exec 9>&-
"$PYTHON" -m pytest -q tests
# All scientific choices are predeclared in config.json and the explicit dataset argument.
# Reuse --run-dir after interruption. Never edit config/code inside an ongoing run.
exec "$PYTHON" -m llada_study.run "$@"
