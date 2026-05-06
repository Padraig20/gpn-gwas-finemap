"""Bootstrap external dependencies that aren't available on PyPI:

  - PolyFun (Python repo, used as a subprocess target).
  - FINEMAP v1.4.1 Linux x86_64 static binary.
  - UCSC liftover chain hg19 <-> hg38 (only the direction we actually use).

All downloads are idempotent: existing files / clones are reused unless
``overwrite`` is requested.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

from ..config import Config


POLYFUN_REPO_URL = "https://github.com/omerwe/polyfun.git"
FINEMAP_TGZ_URL = "http://christianbenner.com/finemap_v1.4.1_x86_64.tgz"
FINEMAP_BINARY_NAME = "finemap_v1.4.1_x86_64"
CHAIN_HG19_TO_HG38 = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz"
)
CHAIN_HG38_TO_HG19 = (
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/liftOver/hg38ToHg19.over.chain.gz"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None
        with (
            tmp.open("wb") as fh,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                leave=False,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))
    tmp.rename(dest)


def _clone_polyfun(target: Path) -> None:
    if (target / ".git").exists():
        print(f"[setup] PolyFun already cloned at {target}; pulling latest.")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=False)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[setup] Cloning PolyFun -> {target}")
    subprocess.run(
        ["git", "clone", "--depth", "1", POLYFUN_REPO_URL, str(target)],
        check=True,
    )


def _install_finemap(bin_dir: Path, exe_path: Path) -> None:
    if exe_path.exists():
        print(f"[setup] FINEMAP already installed at {exe_path}.")
        return
    bin_dir.mkdir(parents=True, exist_ok=True)
    tgz = bin_dir / "finemap_v1.4.1_x86_64.tgz"
    if not tgz.exists():
        print(f"[setup] Downloading FINEMAP v1.4.1 -> {tgz}")
        _download(FINEMAP_TGZ_URL, tgz)
    print(f"[setup] Extracting {tgz.name}")
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(bin_dir)
    extracted_dir = bin_dir / FINEMAP_BINARY_NAME
    extracted_bin = extracted_dir / FINEMAP_BINARY_NAME
    if not extracted_bin.exists():
        candidates = list(extracted_dir.glob("finemap*"))
        if not candidates:
            raise RuntimeError(
                f"Could not find FINEMAP binary inside {extracted_dir}"
            )
        extracted_bin = candidates[0]
    shutil.copy2(extracted_bin, exe_path)
    exe_path.chmod(exe_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"[setup] Installed FINEMAP -> {exe_path}")


def _install_chains(reference_dir: Path) -> None:
    reference_dir.mkdir(parents=True, exist_ok=True)
    for url, name in (
        (CHAIN_HG19_TO_HG38, "hg19ToHg38.over.chain.gz"),
        (CHAIN_HG38_TO_HG19, "hg38ToHg19.over.chain.gz"),
    ):
        dest = reference_dir / name
        if dest.exists():
            print(f"[setup] Chain {name} already present.")
            continue
        print(f"[setup] Downloading {name} -> {dest}")
        _download(url, dest)


def _verify(cfg: Config) -> None:
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    finemap_exe = cfg.paths.absolute("finemap_exe")
    reference_dir = cfg.paths.absolute("reference_dir")

    finemapper_py = polyfun_dir / "finemapper.py"
    if not finemapper_py.exists():
        raise RuntimeError(f"Missing PolyFun finemapper.py at {finemapper_py}")

    if not finemap_exe.exists() or not os.access(finemap_exe, os.X_OK):
        raise RuntimeError(f"FINEMAP binary not executable at {finemap_exe}")

    chain = reference_dir / "hg19ToHg38.over.chain.gz"
    if not chain.exists():
        raise RuntimeError(f"Missing chain file at {chain}")

    print("[setup] Verification passed.")


def run_setup(
    cfg: Config,
    *,
    skip_polyfun: bool = False,
    skip_finemap: bool = False,
    skip_chain: bool = False,
) -> None:
    """End-to-end bootstrap of all external dependencies."""
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    finemap_exe = cfg.paths.absolute("finemap_exe")
    bin_dir = finemap_exe.parent
    reference_dir = cfg.paths.absolute("reference_dir")

    if not skip_polyfun:
        _clone_polyfun(polyfun_dir)
    if not skip_finemap:
        _install_finemap(bin_dir, finemap_exe)
    if not skip_chain:
        _install_chains(reference_dir)

    _verify(cfg)
