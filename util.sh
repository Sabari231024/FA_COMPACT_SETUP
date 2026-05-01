#!/usr/bin/env bash
set -e

LOG_FILE="setup.log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE"
}

run() {
    log "\n[CMD] $*"
    "$@"
}

# =========================
# INSTALL PYTHON
# =========================
install_python() {
    PY="$1"

    log "[INFO] Installing Python $PY"

    apt-get update
    apt-get install -y software-properties-common curl

    add-apt-repository -y ppa:deadsnakes/ppa
    apt-get update

    run apt-get install -y \
        python${PY} \
        python${PY}-dev \
        python${PY}-distutils

    ln -sf /usr/bin/python${PY} /usr/bin/python

    # install pip
    curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    run python get-pip.py
}

# =========================
# INSTALL TORCH
# =========================
install_torch() {
    PY_CMD="$1"
    TORCH="$2"
    CUDA="$3"

    CUDA_TAG="cu$(echo $CUDA | tr -d '.')"

    log "[INFO] Installing PyTorch $TORCH (CUDA $CUDA)"

    run "$PY_CMD" -m pip install --upgrade pip

    run "$PY_CMD" -m pip install \
        torch==$TORCH \
        torchvision \
        torchaudio \
        --index-url https://download.pytorch.org/whl/${CUDA_TAG}
}

# =========================
# INSTALL FLASH ATTENTION
# =========================
install_fa() {
    PY_CMD="$1"
    URL="$2"

    WHEEL=$(basename "$URL")

    log "[INFO] Downloading FlashAttention"
    run wget -O "$WHEEL" "$URL"

    log "[INFO] Installing FlashAttention"
    run "$PY_CMD" -m pip install --no-deps "$WHEEL"

    rm "$WHEEL"
}

# =========================
# SET DEFAULT PYTHON
# =========================
set_default_python() {
    PY_CMD="$1"

    PY_BIN=$(dirname "$(which $PY_CMD)")

    export PATH="$PY_BIN:$PATH"
    hash -r

    log "[INFO] Default python set to $PY_CMD"
}

# =========================
# MAIN
# =========================
main() {
    : > "$LOG_FILE"

    if [ "$#" -lt 4 ]; then
        echo "Usage: util.sh <python> <torch> <cuda> <fa_url>"
        exit 1
    fi

    PYTHON_VERSION="$1"
    TORCH_VERSION="$2"
    CUDA_VERSION="$3"
    FA_URL="$4"

    log "===== INPUT CONFIG ====="
    log "Python: $PYTHON_VERSION"
    log "Torch : $TORCH_VERSION"
    log "CUDA  : $CUDA_VERSION"
    log "FA URL: $FA_URL"
    log "========================"

    install_python "$PYTHON_VERSION"

    PY_CMD="python${PYTHON_VERSION}"

    install_torch "$PY_CMD" "$TORCH_VERSION" "$CUDA_VERSION"
    install_fa "$PY_CMD" "$FA_URL"

    set_default_python "$PY_CMD"

    log "\n===== FINAL CHECK ====="
    run python --version
    run python -c "import torch; print('Torch:', torch.__version__)"
    run python -c "import torch; print('CUDA:', torch.version.cuda)"
    run python -c "import flash_attn; print('FlashAttention OK')"

    log "\nINSTALLATION COMPLETE"
}

main "$@"
