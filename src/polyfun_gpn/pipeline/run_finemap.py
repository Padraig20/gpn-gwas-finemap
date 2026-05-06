"""Run FINEMAP via PolyFun's ``finemapper.py`` for one or many loci.

Per locus we shell out to PolyFun's wrapper script with:

  python <polyfun>/finemapper.py \\
      --method finemap --finemap-exe <path> \\
      --ld <UKB-S3 URL prefix>chr{C}_{S}_{E} \\
      --sumstats output/loci/{prior}/{id}/sumstats.gz \\
      --n <Neff> --chr {C} --start {S} --end {E} \\
      --max-num-causal {K} --memory {GB} --threads {T} \\
      --cache-dir data/ld_cache \\
      --out output/loci/{prior}/{id}/finemap.gz

For genome-wide runs we delegate to PolyFun's ``create_finemapper_jobs.py``
which already partitions the genome into 3 Mb regions and emits one command
per region; we add ``--pvalue-cutoff`` so only signals are run, then execute
with bounded concurrency.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from ..config import Config
from ..loci.demo import load_loci
from ..loci.select import UKBBlock, snap_to_ukb_blocks


def _polyfun_python_args(cfg: Config) -> tuple[str, dict[str, str]]:
    """Return the python interpreter + env to use when running PolyFun.

    PolyFun expects to be importable from its own root, so we add it to
    ``PYTHONPATH``. We use the project's currently-running interpreter
    (``sys.executable``) since UV will already have placed all of PolyFun's
    Python deps into the venv via our pyproject.
    """
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{polyfun_dir}{os.pathsep}{env.get('PYTHONPATH', '')}".strip(os.pathsep)
    )
    return sys.executable, env


def _build_finemap_cmd(
    cfg: Config,
    *,
    locus_id: str,
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
    ld_prefix = cfg.finemap.ukb_ld_url_prefix.rstrip("/") + "/" + block.url_suffix

    cmd = [
        sys.executable,
        str(polyfun_dir / "finemapper.py"),
        "--method", cfg.finemap.method,
        "--finemap-exe", str(finemap_exe),
        "--ld", ld_prefix,
        "--sumstats", str(sumstats_path),
        "--n", str(n_eff),
        "--chr", str(chrom),
        "--start", str(block.start),
        "--end", str(block.end),
        "--max-num-causal", str(cfg.finemap.max_num_causal),
        "--memory", str(cfg.finemap.memory_gb),
        "--threads", str(cfg.finemap.threads),
        "--cache-dir", str(ld_cache),
        "--out", str(out_path),
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

    pairs = snap_to_ukb_blocks(load_loci(loci_path))
    py_exe, env = _polyfun_python_args(cfg)
    _ = py_exe  # we always use sys.executable below; env is the important part

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
            locus_id=locus.locus_id,
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


def run_all(cfg: Config, *, prior_mode: str, chrom: str | None) -> None:
    """Genome-wide fine-mapping via PolyFun's create_finemapper_jobs.py.

    The flow:
      1. Generate per-locus PolyFun sumstats once using the *whole* sumstats
         file's SNPVAR column (re-using ``prepare_loci`` would require us to
         pre-define every locus, which is exactly what
         ``create_finemapper_jobs.py`` does for us). Instead we write a
         single genome-wide sumstats parquet with SNPVAR attached and feed
         that to ``create_finemapper_jobs.py``.
      2. Run the generated commands with bounded concurrency.

    The resulting per-locus output files are written under
    ``output/genome_wide/{prior_mode}/`` to keep them separate from the
    demo-locus runs.
    """
    polyfun_dir = cfg.paths.absolute("polyfun_dir")
    create_jobs = polyfun_dir / "create_finemapper_jobs.py"
    if not create_jobs.exists():
        raise RuntimeError(
            f"{create_jobs} missing; run `polyfun-gpn setup` first."
        )

    finemap_exe = cfg.paths.absolute("finemap_exe")
    if not finemap_exe.exists():
        raise RuntimeError("FINEMAP binary missing; run `polyfun-gpn setup`.")

    gw_dir = cfg.paths.absolute("output_dir") / "genome_wide" / prior_mode
    gw_dir.mkdir(parents=True, exist_ok=True)
    sumstats_gw = _build_genome_wide_sumstats(cfg, prior_mode=prior_mode, dest_dir=gw_dir)
    n_median = _median_n_for_locus(sumstats_gw)

    out_prefix = gw_dir / "fm"
    jobs_file = gw_dir / "jobs.txt"

    cmd = [
        sys.executable,
        str(create_jobs),
        "--sumstats", str(sumstats_gw),
        "--n", str(n_median),
        "--method", cfg.finemap.method,
        "--max-num-causal", str(cfg.finemap.max_num_causal),
        "--out-prefix", str(out_prefix),
        "--jobs-file", str(jobs_file),
        "--pvalue-cutoff", str(cfg.finemap.pvalue_cutoff),
        "--memory", str(cfg.finemap.memory_gb),
        "--python3", sys.executable,
    ]
    if chrom is not None:
        cmd += ["--chr", str(chrom)]

    py_exe, env = _polyfun_python_args(cfg)
    _ = py_exe
    print("[run-all] Generating per-region jobs ...")
    print("$ " + " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, env=env, check=True)

    job_lines = [l for l in jobs_file.read_text().splitlines() if l.strip()]
    if not job_lines:
        print("[run-all] No genome-wide jobs were produced.")
        return

    # Inject our FINEMAP binary path into each generated command, since
    # create_finemapper_jobs.py doesn't propagate --finemap-exe.
    augmented = []
    for line in job_lines:
        line = line.strip()
        if cfg.finemap.method == "finemap" and "--finemap-exe" not in line:
            line += f" --finemap-exe {shlex.quote(str(finemap_exe))}"
        if "--cache-dir" not in line:
            line += f" --cache-dir {shlex.quote(str(cfg.paths.absolute('ld_cache')))}"
        if "--allow-missing" not in line:
            line += " --allow-missing"
        augmented.append(line)
    jobs_file.write_text("\n".join(augmented) + "\n")

    print(f"[run-all] {len(augmented)} region jobs queued.")
    max_workers = max(1, cfg.finemap.max_concurrent_jobs)

    def _execute(idx_line):
        idx, line = idx_line
        log_path = gw_dir / f"region_{idx:05d}.log"
        with log_path.open("w") as log:
            log.write("$ " + line + "\n\n")
            log.flush()
            proc = subprocess.run(line, env=env, shell=True, stdout=log, stderr=subprocess.STDOUT)
        return idx, proc.returncode, log_path

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_execute, (i, l)) for i, l in enumerate(augmented)]
        for fut in as_completed(futures):
            idx, rc, log_path = fut.result()
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"[run-all] region {idx}: {status}  log -> {log_path}")


def _build_genome_wide_sumstats(
    cfg: Config, *, prior_mode: str, dest_dir: Path
) -> Path:
    """Materialize a genome-wide sumstats file with SNPVAR attached.

    For ``prior_mode="entropy"`` we look up entropy for every variant, which
    requires scanning every entropy parquet. For ``"uniform"`` and
    ``"none"`` it's a simple copy (with or without SNPVAR).
    """
    out_file = dest_dir / f"sumstats.{prior_mode}.tsv"
    if out_file.exists():
        print(f"[run-all] Reusing {out_file}")
        return out_file
    src = cfg.paths.absolute("gwas_harmonised")
    if not src.exists():
        raise FileNotFoundError(f"Missing harmonised sumstats at {src}; run `harmonize`.")

    polyfun_cols = ["SNP", "CHR", "BP", "A1", "A2", "Z", "N"]

    if prior_mode == "none":
        pl.scan_parquet(src).select(polyfun_cols).sink_csv(
            out_file, separator="\t", include_header=True
        )
        return out_file

    if prior_mode == "uniform":
        pl.scan_parquet(src).with_columns(
            pl.lit(1.0).alias("SNPVAR")
        ).select(polyfun_cols + ["SNPVAR"]).sink_csv(
            out_file, separator="\t", include_header=True
        )
        return out_file

    # prior_mode == "entropy"
    from ..entropy.background import load_background
    from ..entropy.lookup import lookup_entropy
    from ..entropy.priors import entropy_snpvar
    from ..coords import load_liftover, lift_positions

    density, edges = load_background(cfg)

    # Process per-chromosome to bound memory.
    df = pl.read_parquet(src)
    pieces = []
    for chrom, sub in df.group_by("CHR"):
        chrom_label = chrom[0] if isinstance(chrom, tuple) else chrom
        chroms = [chrom_label] * sub.height
        positions = sub["BP"].to_numpy()
        if cfg.builds.gwas != cfg.builds.entropy:
            lo = load_liftover(
                cfg.paths.absolute("reference_dir"),
                cfg.builds.gwas,
                cfg.builds.entropy,
            )
            new_chrom, new_pos = lift_positions(lo, chroms, positions)
            ent = lookup_entropy(
                cfg.paths.absolute("entropy_dir"),
                list(new_chrom),
                new_pos,
            )
        else:
            ent = lookup_entropy(
                cfg.paths.absolute("entropy_dir"),
                chroms,
                positions,
            )
        snpvar = entropy_snpvar(ent, density, edges, cfg.prior)
        finite = pl.Series("__f__", [bool(x) for x in (snpvar == snpvar)])
        median = float(snpvar[finite.to_numpy()].mean()) if finite.any() else 1.0
        snpvar = pl.Series(
            "SNPVAR",
            [v if v == v else median for v in snpvar.tolist()],
            dtype=pl.Float64,
        )
        pieces.append(sub.with_columns(snpvar))
    out_df = pl.concat(pieces, how="vertical").select(polyfun_cols + ["SNPVAR"])
    out_df.write_csv(out_file, separator="\t", include_header=True)
    return out_file
