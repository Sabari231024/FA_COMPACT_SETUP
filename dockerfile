# =========================
# BASE IMAGE
# =========================
ARG DRIVER_VERSION
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DRIVER_VERSION
ARG PYTHON_VERSION=""
ARG TORCH_VERSION=""
ARG CUDA_VERSION_FILTER=""

ENV DEBIAN_FRONTEND=noninteractive

# =========================
# SYSTEM SETUP
# =========================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    wget curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip
RUN pip install pyyaml requests

WORKDIR /app

# =========================
# COPY PROJECT (OPTIONAL)
# =========================
COPY . /app
RUN mkdir -p /app/Scripts

# =========================
# FETCH YAML
# =========================
RUN wget -O fa_compat.yaml \
    https://raw.githubusercontent.com/Sabari231024/FA_COMPACT_SETUP/main/fa_compat.yaml

# =========================
# RESOLVE + INSTALL
# =========================
RUN python3 - <<EOF
import yaml, subprocess, sys, os, json, urllib.request
from urllib.request import urlretrieve
from urllib.parse import unquote

DRIVER = "${DRIVER_VERSION}"
REQ_PY = "${PYTHON_VERSION}"
REQ_TORCH = "${TORCH_VERSION}"
REQ_CUDA = "${CUDA_VERSION_FILTER}"

print("DEBUG DRIVER =", DRIVER)

# -------------------------
# VERSION UTILS
# -------------------------
def vt(v):
    if v is None:
        return (0,)
    v = str(v).strip()
    if not v:
        return (0,)
    return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))

def geq(a, b):
    if not b:
        return True
    return vt(a) >= vt(b)

# -------------------------
# LOAD YAML
# -------------------------
with open("fa_compat.yaml") as f:
    data = yaml.safe_load(f)["flash_attention"]

valid = [r for r in data if geq(DRIVER, r.get("min_driver"))]

if not valid:
    sys.exit(f"[ERROR] No configs support driver {DRIVER}")

# -------------------------
# OPTIONAL FILTERS
# -------------------------
def match(r):
    if REQ_PY and str(r.get("Python")) != REQ_PY:
        return False
    if REQ_TORCH and str(r.get("PyTorch")) != REQ_TORCH:
        return False
    if REQ_CUDA and str(r.get("CUDA")) != REQ_CUDA:
        return False
    return True

filtered = [r for r in valid if match(r)]

if not filtered:
    sys.exit("[ERROR] No config after applying filters")

# -------------------------
# PICK BEST (UNCHANGED)
# -------------------------
filtered.sort(key=lambda r: (vt(r.get("PyTorch")), vt(r.get("Python"))))
best = filtered[-1]

PYTHON = str(best["Python"])
TORCH = str(best["PyTorch"])
CUDA = str(best["CUDA"])
FA_URL = best["Download URL"]

print("\\n[INFO] Selected config:")
print(best)

# -------------------------
# ROBUST PYTHON RESOLVER
# -------------------------
def resolve_python_full_version(py):
    url = "https://www.python.org/api/v2/downloads/release/?is_published=true"
    data = json.loads(urllib.request.urlopen(url).read())

    if isinstance(data, dict) and "results" in data:
        releases = data["results"]
    else:
        releases = data

    versions = []
    for r in releases:
        name = r.get("name", "")
        if not name:
            continue

        v = name.replace("Python ", "")
        if v.startswith(py + "."):
            versions.append(v)

    if not versions:
        print("[WARN] fallback Python:", py)
        return py + ".0"

    def safe_version_tuple(v):
        parts = []
        for p in v.split("."):
            num = ""
            for c in p:
                if c.isdigit():
                    num += c
                else:
                    break
            parts.append(int(num) if num else 0)
        return tuple(parts)

    versions.sort(key=safe_version_tuple)
    return versions[-1]

full_py = resolve_python_full_version(PYTHON)
py_short = ".".join(full_py.split(".")[:2])

print("[INFO] Resolved Python:", full_py)

# -------------------------
# INSTALL PYTHON FROM SOURCE
# -------------------------
subprocess.check_call(["apt-get", "update"])
subprocess.check_call([
    "apt-get", "install", "-y",
    "build-essential", "libssl-dev", "zlib1g-dev",
    "libncurses5-dev", "libbz2-dev", "libreadline-dev",
    "libsqlite3-dev", "libffi-dev", "liblzma-dev"
])

tar_name = f"Python-{full_py}.tgz"
url = f"https://www.python.org/ftp/python/{full_py}/Python-{full_py}.tgz"

print("[INFO] Downloading Python:", url)
urlretrieve(url, tar_name)

subprocess.check_call(["tar", "-xf", tar_name])
os.chdir(f"Python-{full_py}")

subprocess.check_call(["./configure", "--enable-optimizations"])
subprocess.check_call(["make", "-j2"])
subprocess.check_call(["make", "altinstall"])

# link python
subprocess.check_call([
    "ln", "-sf",
    f"/usr/local/bin/python{py_short}",
    "/usr/bin/python"
])

# pip
subprocess.check_call([
    f"/usr/local/bin/python{py_short}", "-m", "ensurepip"
])

subprocess.check_call([
    "python", "-m", "pip", "install", "--upgrade", "pip"
])

# -------------------------
# INSTALL PYTORCH
# -------------------------
cuda_tag = "cu" + CUDA.replace(".", "")

print("[INFO] Installing Torch:", TORCH)
subprocess.check_call([
    "pip", "install",
    f"torch=={TORCH}",
    "--index-url", f"https://download.pytorch.org/whl/{cuda_tag}"
])

# -------------------------
# INSTALL FLASH ATTENTION
# -------------------------
wheel = unquote(FA_URL.split("/")[-1])

print("[INFO] Downloading FA:", FA_URL)
urlretrieve(FA_URL, wheel)

print("[INFO] Installing FA:", wheel)
subprocess.check_call(["pip", "install", "--no-deps", wheel])

# -------------------------
# SAVE BUILD INFO
# -------------------------
with open("/app/build_info.txt", "w") as f:
    f.write(f"Driver: {DRIVER}\\n")
    f.write(f"Python: {PYTHON}\\n")
    f.write(f"Resolved Python: {full_py}\\n")
    f.write(f"PyTorch: {TORCH}\\n")
    f.write(f"CUDA: {CUDA}\\n")
    f.write(f"FlashAttention URL: {FA_URL}\\n")

EOF

# =========================
# OPTIONAL REQUIREMENTS
# =========================
RUN if [ -f /app/Scripts/requirements.txt ]; then \
    pip install -r /app/Scripts/requirements.txt; \
    fi

# =========================
# RUNTIME
# =========================
ENV RUN_CMD=""

CMD ["bash", "-c", "\
    echo '===== BUILD INFO ====='; \
    cat /app/build_info.txt; \
    echo '======================'; \
    if [ -z \"$RUN_CMD\" ]; then \
    echo 'No RUN_CMD provided. Opening shell...'; \
    bash; \
    else \
    echo \"Running: $RUN_CMD\"; \
    eval \"$RUN_CMD\"; \
    fi"]