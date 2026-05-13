"""Run FINEMAP via PolyFun's ``finemapper.py`` for a list of loci.

Per locus we shell out to PolyFun's wrapper script with::

  python <polyfun>/finemapper.py \\
      --method finemap --finemap-exe <path> \\
      --ld <UKB-S3 URL prefix>chr{C}_{S}_{E} \\
      --sumstats output_dir/loci/{prior}/{id}/sumstats.tsv \\
      --n <Neff> --chr {C} --start {S} --end {E} \\
      --max-num-causal {K} --memory {GB} --threads {T} \\
      --cache-dir data/ld_cache \\
      --out output_dir/loci/{prior}/{id}/finemap.gz

For ``--prior none`` we add ``--non-funct`` so PolyFun runs FINEMAP without a
SNPVAR column. For ``--prior entropy`` PolyFun reads the SNPVAR column from
the per-locus sumstats file.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from ..config import Config, resolve_plink_prefix, validate_finemap_ld
from ..loci.demo import load_loci
from ..loci.select import UKBBlock, snap_to_ukb_blocks


def _polyfun_env(cfg: Config) -> dict[str, str]:
    """Env with PolyFun on ``PYTHONPATH`` (PolyFun expects to import from its root)."""
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{polyfun_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    )
    return env


def _build_finemap_cmd(
    cfg: Config,
    *,
    chrom: str,
    block: UKBBlock,
    sumstats_path: Path,
    n_eff: int,
    out_path: Path,
    prior_mode: str,
) -> list[str]:
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    finemap_exe = cfg.paths.absolute("finemap_exe")
    ld_cache = cfg.paths.absolute("ld_cache")

    cmd: list[str] = [
        sys.executable,
        str(polyfun_dir / "finemapper.py"),
        "--method",
        cfg.finemap.method,
        "--finemap-exe",
        str(finemap_exe),
    ]
    if cfg.finemap.ld_mode == "plink":
        cmd += ["--geno", str(resolve_plink_prefix(cfg))]
    else:
        ld_prefix = cfg.finemap.ld_npz_url_prefix.rstrip("/") + "/" + block.url_suffix
        cmd += ["--ld", ld_prefix]

    cmd += [
        "--sumstats",
        str(sumstats_path),
        "--n",
        str(n_eff),
        "--chr",
        str(chrom),
        "--start",
        str(block.start),
        "--end",
        str(block.end),
        "--max-num-causal",
        str(cfg.finemap.max_num_causal),
        "--memory",
        str(cfg.finemap.memory_gb),
        "--threads",
        str(cfg.finemap.threads),
        "--cache-dir",
        str(ld_cache),
        "--out",
        str(out_path),
        "--allow-missing",
    ]
    if prior_mode == "none":
        cmd.append("--non-funct")
    return cmd


def _median_n_for_locus(sumstats_path: Path) -> int:
    df = pl.read_csv(sumstats_path, separator="\t").select("N")
    return int(df["N"].median())


def run_loci(cfg: Config, *, loci_path: Path, prior_mode: str) -> None:
    """Run FINEMAP for each locus in a TSV (already prepared)."""
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    if not (polyfun_dir / "finemapper.py").exists():
        raise RuntimeError(
            f"finemapper.py not found under {polyfun_dir}; run `polyfun-gpn setup`."
        )
    if not cfg.paths.absolute("finemap_exe").exists():
        raise RuntimeError(
            "FINEMAP binary missing; run `polyfun-gpn setup` to install it."
        )
    cfg.paths.absolute("ld_cache").mkdir(parents=True, exist_ok=True)
    validate_finemap_ld(cfg)

    pairs = snap_to_ukb_blocks(load_loci(loci_path))
    env = _polyfun_env(cfg)

    loci_root = cfg.paths.absolute("output_dir") / "loci" / prior_mode
    tasks = []
    for locus, block in pairs:
        out_dir = loci_root / locus.locus_id
        sumstats_path = out_dir / "sumstats.tsv"
        out_path = out_dir / "finemap.gz"
        if not sumstats_path.exists():
            print(
                f"[run] {locus.locus_id}: missing {sumstats_path}; "
                "did you run `prepare-loci`?"
            )
            continue
        n_eff = _median_n_for_locus(sumstats_path)
        cmd = _build_finemap_cmd(
            cfg,
            chrom=locus.chrom,
            block=block,
            sumstats_path=sumstats_path,
            n_eff=n_eff,
            out_path=out_path,
            prior_mode=prior_mode,
        )
        tasks.append((locus.locus_id, cmd, out_dir))

    if not tasks:
        print("[run] No tasks to execute.")
        return

    max_workers = max(1, cfg.finemap.max_concurrent_jobs)
    print(
        f"[run] Running {len(tasks)} FINEMAP jobs with up to "
        f"{max_workers} workers."
    )

    def _execute(task):
        locus_id, cmd, out_dir = task
        log_path = out_dir / "finemap.log"
        with log_path.open("w") as log:
            log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        return locus_id, proc.returncode, log_path

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_execute, t) for t in tasks]
        for fut in as_completed(futures):
            locus_id, rc, log_path = fut.result()
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"[run] {locus_id}: {status}  log -> {log_path}")
