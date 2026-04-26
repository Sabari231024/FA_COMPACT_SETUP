#!/usr/bin/env bash
set -e

GITHUB_YAML_URL="https://raw.githubusercontent.com/Sabari231024/FA_COMPACT_SETUP/main/fa_compat.yaml"
CACHE_FILE="fa_cache.yaml"
CACHE_TTL=3600
LOG_FILE="setup.log"

log() {
    echo -e "$1" | tee -a "$LOG_FILE" >&2
}

run() {
    log "\n[CMD] $*"
    "$@"
}

# ✅ version comparison (robust)
version_geq() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]
}

get_driver_version() {
    nvidia-smi | grep "Driver Version" | awk '{print $3}'
}

fetch_yaml() {
    curl -f -s "$GITHUB_YAML_URL" -o "$CACHE_FILE" || {
        log "[ERROR] Failed to download YAML"
        exit 1
    }
}

load_yaml() {
    if [ -f "$CACHE_FILE" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$CACHE_FILE") ))
        if [ "$age" -lt "$CACHE_TTL" ]; then
            return
        fi
    fi
    fetch_yaml
}

# ✅ robust YAML parser
parse_yaml() {
    awk '
    BEGIN {
        python=""; torch=""; driver=""; cuda=""; url=""
    }

    /^- / {
        if (python != "" && torch != "" && driver != "" && url != "") {
            print python "|" torch "|" driver "|" cuda "|" url
        }
        python=""; torch=""; driver=""; cuda=""; url=""
    }

    /Python:/ { python=$2 }
    /PyTorch:/ { torch=$2 }
    /min_driver:/ { driver=$2 }
    /CUDA:/ { cuda=$2 }
    /Download URL:/ { url=$3 }

    END {
        if (python != "" && torch != "" && driver != "" && url != "") {
            print python "|" torch "|" driver "|" cuda "|" url
        }
    }
    ' "$CACHE_FILE"
}

# ✅ select latest compatible
select_best_config() {
    DRIVER="$1"

    parse_yaml | while IFS="|" read -r py torch min_driver cuda url; do
        log "[DEBUG] Checking: $DRIVER >= $min_driver"

        if version_geq "$DRIVER" "$min_driver"; then
            log "[DEBUG] PASS → Python=$py Torch=$torch"
            echo "$py|$torch|$cuda|$url"
        fi
    done | sort -t'|' -k1,1V -k2,2V | tail -n1
}

# ✅ find correct python binary
find_python() {
    TARGET="$1"

    for py in python$TARGET python${TARGET} python3; do
        if command -v "$py" >/dev/null 2>&1; then
            ver=$($py -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [ "$ver" = "$TARGET" ]; then
                echo "$py"
                return
            fi
        fi
    done

    echo ""
}

install_torch() {
    PY_CMD="$1"
    TORCH="$2"

    log "[INFO] Installing PyTorch using $PY_CMD"
    run "$PY_CMD" -m pip install --upgrade pip
    run "$PY_CMD" -m pip install "torch==$TORCH" torchvision torchaudio
}

install_fa() {
    PY_CMD="$1"
    URL="$2"

    WHEEL=$(basename "$URL")

    log "[INFO] Downloading FlashAttention..."
    wget -O "$WHEEL" "$URL"

    run "$PY_CMD" -m pip install --no-deps "$WHEEL"

    rm "$WHEEL"
}

# ✅ CLEAN + PERSISTENT PYTHON SWITCH
set_default_python() {
    PY_CMD="$1"

    PY_BIN=$(dirname "$(which $PY_CMD)")
    CONFIG_FILE="$HOME/.python_default"

    log "[INFO] Setting default python to $PY_CMD"

    # remove old config references
    sed -i "\|$CONFIG_FILE|d" ~/.bashrc 2>/dev/null || true
    sed -i "\|$CONFIG_FILE|d" ~/.bash_profile 2>/dev/null || true

    # create clean config
    cat > "$CONFIG_FILE" <<EOF
# Auto-generated python default
export PATH=$PY_BIN:\$PATH
EOF

    # attach cleanly
    echo "source $CONFIG_FILE" >> ~/.bashrc
    echo "source $CONFIG_FILE" >> ~/.bash_profile

    # apply immediately
    export PATH="$PY_BIN:$PATH"
    hash -r

    log "[INFO] Python default set persistently"
}

main() {
    : > "$LOG_FILE"

    DRIVER=$(get_driver_version)
    log "[INFO] Driver: $DRIVER"

    load_yaml

    CONFIG=$(select_best_config "$DRIVER")
    log "[DEBUG] Raw CONFIG: $CONFIG"

    if [ -z "$CONFIG" ]; then
        log "[ERROR] No compatible config found"
        exit 1
    fi

    PY=$(echo "$CONFIG" | cut -d'|' -f1)
    TORCH=$(echo "$CONFIG" | cut -d'|' -f2)
    CUDA=$(echo "$CONFIG" | cut -d'|' -f3)
    URL=$(echo "$CONFIG" | cut -d'|' -f4)

    log "[INFO] Selected:"
    log "       Python: $PY"
    log "       Torch : $TORCH"
    log "       CUDA  : $CUDA"

    PY_CMD=$(find_python "$PY")

    if [ -z "$PY_CMD" ]; then
        log "[ERROR] Python $PY not found on system"
        exit 1
    fi

    log "[INFO] Using interpreter: $PY_CMD"

    install_torch "$PY_CMD" "$TORCH"
    install_fa "$PY_CMD" "$URL"

    # 🔥 persistent python switch
    set_default_python "$PY_CMD"

    log "\n✅ SUCCESS"

    log "\nVerify:"
    run python --version
    run python -c "import torch; print(torch.__version__, torch.version.cuda)"
}

main
