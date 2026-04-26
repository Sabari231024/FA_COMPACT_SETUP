import argparse
import subprocess
import sys
import yaml
import requests
import os
import time
import re
from urllib.request import urlretrieve
from urllib.parse import unquote


GITHUB_YAML_URL = "https://raw.githubusercontent.com/Sabari231024/FA_COMPACT_SETUP/main/fa_compat.yaml"
CACHE_FILE = "fa_cache.yaml"
CACHE_TTL = 3600
LOG_FILE = "setup.log"
PROJECT_ENV = "PROJECT_ENV"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run(cmd):
    log(f"\n[CMD] {' '.join(cmd)}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    return process.returncode


def version_geq(v1, v2):
    return tuple(map(int, v1.split("."))) >= tuple(map(int, v2.split(".")))

def ensure_micromamba():
    home = os.path.expanduser("~")
    bin_dir = os.path.join(home, "micromamba_bin")
    os.makedirs(bin_dir, exist_ok=True)
    micromamba_path = os.path.join(bin_dir, "micromamba")
    if os.path.exists(micromamba_path):
        log("[INFO] micromamba already installed")
        return micromamba_path
    log("[INFO] Installing micromamba...")
    url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
    archive = "micromamba.tar.bz2"
    run(["wget", "-O", archive, url])
    run(["tar", "-xvjf", archive])
    run(["mv", "bin/micromamba", micromamba_path])

    return micromamba_path

def get_driver_version():
    output = subprocess.check_output(["nvidia-smi"], text=True)

    match = re.search(r"Driver Version:\s+(\d+\.\d+\.\d+)", output)
    if not match:
        sys.exit("[ERROR] Could not parse driver version")

    return match.group(1)

def fetch_yaml():
    r = requests.get(GITHUB_YAML_URL, timeout=10)
    r.raise_for_status()
    return yaml.safe_load(r.text)

def load_yaml():
    if os.path.exists(CACHE_FILE):
        age = time.time() - os.path.getmtime(CACHE_FILE)
        if age < CACHE_TTL:
            with open(CACHE_FILE) as f:
                return yaml.safe_load(f)

    data = fetch_yaml()
    with open(CACHE_FILE, "w") as f:
        yaml.dump(data, f)
    return data

def filter_by_driver(data, driver):
    return [
        row for row in data["flash_attention"]
        if version_geq(driver, row["min_driver"])
    ]

def resolve_fa(data, python, torch, driver):
    candidates = filter_by_driver(data, driver)

    for row in candidates:
        if str(row["Python"]) == str(python) and str(row["PyTorch"]) == str(torch):
            return row

    return candidates[-1]

def create_env(mamba, python, torch):
    root = os.path.expanduser("~/micromamba")

    cmd = [
        mamba, "-r", root,
        "create", "-y",
        "-n", PROJECT_ENV,
        f"python={python}",
        f"pytorch={torch}.*",
        "pip",
        "-c", "conda-forge",
        "-c", "pytorch",
        "-c", "nvidia"
    ]

    if run(cmd) != 0:
        sys.exit("[ERROR] Environment creation failed")

def install_fa(mamba, config):
    root = os.path.expanduser("~/micromamba")

    url = config["Download URL"]
    wheel = unquote(url.split("/")[-1])

    log("[INFO] Installing Flash Attention...")
    urlretrieve(url, wheel)

    cmd = [
        mamba, "-r", root,
        "run", "-n", PROJECT_ENV,
        "pip", "install", "--no-deps", wheel
    ]

    if run(cmd) != 0:
        sys.exit("[ERROR] Flash Attention install failed")

    os.remove(wheel)

def main():
    open(LOG_FILE, "w").close()

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="3.11")
    parser.add_argument("--torch", default="2.2")

    args, _ = parser.parse_known_args()

    mamba = ensure_micromamba()
    driver = get_driver_version()

    log(f"[INFO] Driver: {driver}")
    log(f"[INFO] Requested Torch: {args.torch}")

    data = load_yaml()
    config = resolve_fa(data, args.python, args.torch, driver)

    python = config["Python"]
    torch = config["PyTorch"]

    log(f"[INFO] Using Python: {python}")
    log(f"[INFO] Using Torch: {torch}")
    log(f"[INFO] Using CUDA (FA): {config['CUDA']}")

    create_env(mamba, python, torch)
    install_fa(mamba, config)

    root = os.path.expanduser("~/micromamba")

    log("\nSUCCESS: Environment ready")

    log("\nTo use environment:")
    log(f"export PATH={os.path.expanduser('~/micromamba_bin')}:$PATH")
    log(f"export MAMBA_ROOT_PREFIX={root}")
    log(f"{mamba} -r {root} run -n {PROJECT_ENV} bash")

    log("\nOr run directly:")
    log(f"{mamba} -r {root} run -n {PROJECT_ENV} python script.py")

    log("\nVerify:")
    run([
        mamba, "-r", root, "run", "-n", PROJECT_ENV,
        "python", "-c",
        "import torch; print(torch.__version__, torch.version.cuda)"
    ])

if __name__ == "__main__":
    main()
