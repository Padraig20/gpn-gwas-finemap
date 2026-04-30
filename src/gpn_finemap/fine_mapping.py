"""End-to-end SuSiE and FINEMAP runs with entropy-derived priors."""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from gpn_finemap.finngen import download_file
from gpn_finemap.priors import add_entropy_prior_columns

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FineMappingRunConfig:
    annotated_variants: Path
    output_dir: Path
    ld_bcor_dir: Path | None = None
    ld_matrix_dir: Path | None = None
    ldstore_exe: str = "ldstore"
    rscript_exe: str = "Rscript"
    finemap_exe: str = "finemap"
    source_method: str = "SUSIE"
    constrained_direction: str = "low"
    prior_method: str = "softmax"
    temperature: float = 1.0
    prior_floor: float = 1e-6
    missing_policy: str = "median"
    finemap_expected_causal_per_region: float = 1.0
    max_causal: int = 10
    n_samples: int = 3775
    max_regions: int | None = None
    max_variants: int = 5000
    run_susie: bool = True
    run_finemap: bool = True
    allow_identity_ld: bool = False
    download_ld_bcor: bool = False


def run_fine_mapping(config: FineMappingRunConfig) -> pl.DataFrame:
    """Prepare LD and launch matched uniform/entropy-prior SuSiE/FINEMAP jobs."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(config.annotated_variants)
    frame = _filter_source_method(frame, config.source_method)
    priors = add_entropy_prior_columns(
        frame,
        constrained_direction=config.constrained_direction,
        prior_method=config.prior_method,
        temperature=config.temperature,
        prior_floor=config.prior_floor,
        missing_policy=config.missing_policy,
        finemap_expected_causal_per_region=config.finemap_expected_causal_per_region,
    )

    jobs: list[dict[str, object]] = []
    _write_susie_runner(config.output_dir / "run_susie_region.R")
    for idx, (region, group) in enumerate(priors.partition_by("region", as_dict=True, maintain_order=True).items()):
        if config.max_regions is not None and idx >= config.max_regions:
            break
        region_text = region[0] if isinstance(region, tuple) else str(region)
        region_info = parse_region(region_text, group)
        clean = _prepare_region_variants(group, config.max_variants)
        if clean.height < 2:
            logger.warning("Skipping %s because it has fewer than two usable variants", region_text)
            continue

        region_dir = config.output_dir / "regions" / _safe_stem(region_text)
        region_dir.mkdir(parents=True, exist_ok=True)
        ld_path = _resolve_ld_matrix(config, region_info, clean, region_dir)

        entropy_inputs = _write_method_inputs(clean, region_dir, "entropy", config)
        uniform_inputs = _write_method_inputs(_with_uniform_priors(clean), region_dir, "uniform", config)

        if config.run_susie:
            _run_susie(config, entropy_inputs, ld_path, region_dir / "susie_entropy")
            _run_susie(config, uniform_inputs, ld_path, region_dir / "susie_uniform")
        if config.run_finemap:
            _run_finemap(config, entropy_inputs, ld_path, region_dir / "finemap_entropy")
            _run_finemap(config, uniform_inputs, ld_path, region_dir / "finemap_uniform")

        jobs.append(
            {
                "region": region_text,
                "chrom": region_info["chrom"],
                "start": region_info["start"],
                "end": region_info["end"],
                "n_variants": clean.height,
                "ld_path": str(ld_path),
                "entropy_z_path": str(entropy_inputs["finemap_z"]),
                "uniform_z_path": str(uniform_inputs["finemap_z"]),
                "susie_entropy_out": str(region_dir / "susie_entropy.pip.tsv"),
                "susie_uniform_out": str(region_dir / "susie_uniform.pip.tsv"),
                "finemap_entropy_dir": str(region_dir / "finemap_entropy"),
                "finemap_uniform_dir": str(region_dir / "finemap_uniform"),
            }
        )

    manifest = pl.DataFrame(jobs)
    manifest_path = config.output_dir / "fine_mapping_manifest.tsv"
    if not manifest.is_empty():
        manifest.write_csv(manifest_path, separator="\t")
    logger.info("Wrote fine-mapping manifest to %s", manifest_path)
    return manifest


def parse_region(region: str, group: pl.DataFrame) -> dict[str, int | str]:
    """Parse FinnGen-style region text, falling back to min/max positions."""

    match = re.search(r"(?:chr)?(?P<chrom>[0-9XY]+)[:_](?P<start>[0-9]+)[-_](?P<end>[0-9]+)", region)
    if match:
        return {
            "chrom": match.group("chrom").upper(),
            "start": int(match.group("start")),
            "end": int(match.group("end")),
        }
    chrom = str(group.get_column("chrom")[0]).removeprefix("chr").upper()
    return {
        "chrom": chrom,
        "start": int(group.get_column("pos").min()),
        "end": int(group.get_column("pos").max()),
    }


def _filter_source_method(frame: pl.DataFrame, source_method: str) -> pl.DataFrame:
    if "method" not in frame.columns:
        return frame
    filtered = frame.filter(pl.col("method") == source_method.upper())
    if filtered.is_empty():
        raise ValueError(f"No rows found for source method {source_method!r}")
    return filtered


def _prepare_region_variants(group: pl.DataFrame, max_variants: int) -> pl.DataFrame:
    z_expr = _z_expression(group)
    prepared = (
        group.with_columns(z_expr.alias("z"))
        .filter(pl.col("z").is_not_null() & pl.col("variant_id").is_not_null())
        .unique(subset=["variant_id"], keep="first")
        .sort("pos")
    )
    if prepared.height > max_variants:
        logger.warning("Trimming region from %d to %d variants by smallest p/PIP", prepared.height, max_variants)
        sort_col = "p" if "p" in prepared.columns else "pval" if "pval" in prepared.columns else "pip"
        prepared = prepared.sort(sort_col, descending=(sort_col == "pip")).head(max_variants).sort("pos")
    return prepared


def _z_expression(group: pl.DataFrame) -> pl.Expr:
    if "z" in group.columns:
        return pl.col("z").cast(pl.Float64)
    se_col = "se" if "se" in group.columns else "sebeta" if "sebeta" in group.columns else None
    if "beta" not in group.columns or se_col is None:
        raise ValueError("Need either z or beta plus se/sebeta columns to run fine-mapping")
    return pl.when(pl.col(se_col) != 0).then(pl.col("beta") / pl.col(se_col)).otherwise(None)


def _with_uniform_priors(group: pl.DataFrame) -> pl.DataFrame:
    n = group.height
    uniform = np.repeat(1.0 / n, n)
    return group.with_columns(
        pl.Series("susie_prior_weight", uniform),
        pl.Series("finemap_prior_probability", uniform),
        pl.Series("SNPVAR", uniform),
    )


def _write_method_inputs(
    group: pl.DataFrame,
    region_dir: Path,
    label: str,
    config: FineMappingRunConfig,
) -> dict[str, Path]:
    input_dir = region_dir / label
    input_dir.mkdir(parents=True, exist_ok=True)
    susie_z = input_dir / f"{label}.susie.tsv"
    finemap_z = input_dir / f"{label}.finemap.z"
    prior_path = input_dir / f"{label}.prior_weights.tsv"

    group.select("variant_id", "z", "susie_prior_weight").write_csv(susie_z, separator="\t")
    group.select("variant_id", "susie_prior_weight").write_csv(prior_path, separator="\t")

    id_col = "rsid" if "rsid" in group.columns else "variant_id"
    columns = [
        pl.col(id_col).cast(pl.Utf8).alias("rsid"),
        pl.col("chrom").alias("chromosome"),
        pl.col("pos").alias("position"),
        pl.col("ref").alias("allele1"),
        pl.col("alt").alias("allele2"),
        _optional_or_default(group, "maf", 0.1).alias("maf"),
        pl.col("beta").cast(pl.Float64).alias("beta"),
        _optional_or_default(group, "se", None, fallback="sebeta").alias("se"),
        pl.col("finemap_prior_probability").alias("prob"),
    ]
    group.select(columns).write_csv(finemap_z, separator=" ")
    return {"susie_z": susie_z, "finemap_z": finemap_z, "prior": prior_path}


def _optional_or_default(group: pl.DataFrame, column: str, default: float | None, fallback: str | None = None) -> pl.Expr:
    if column in group.columns:
        return pl.col(column)
    if fallback and fallback in group.columns:
        return pl.col(fallback)
    if default is None:
        raise ValueError(f"Missing required column {column!r}")
    return pl.lit(default)


def _resolve_ld_matrix(
    config: FineMappingRunConfig,
    region_info: dict[str, int | str],
    variants: pl.DataFrame,
    region_dir: Path,
) -> Path:
    stem = _safe_stem(f"chr{region_info['chrom']}_{region_info['start']}_{region_info['end']}")
    if config.ld_matrix_dir:
        candidate = config.ld_matrix_dir / f"{stem}.ld"
        if candidate.exists():
            logger.info("Using precomputed LD matrix: %s", candidate)
            return candidate

    if config.ld_bcor_dir:
        table_path = region_dir / f"{stem}.ldstore.table"
        ld_path = region_dir / f"{stem}.ld"
        bcor = _bcor_path(config.ld_bcor_dir, str(region_info["chrom"]), download=config.download_ld_bcor)
        _run_ldstore(config.ldstore_exe, bcor, int(region_info["start"]), int(region_info["end"]), table_path)
        _ldstore_table_to_matrix(table_path, variants, ld_path)
        return ld_path

    if config.allow_identity_ld:
        ld_path = region_dir / f"{stem}.identity.ld"
        logger.warning("Writing identity LD matrix for %s; use only for smoke tests", stem)
        _write_ld_matrix(np.eye(variants.height), ld_path)
        return ld_path

    raise ValueError("Provide --ld-bcor-dir, --ld-matrix-dir, or --allow-identity-ld")


def _bcor_path(ld_bcor_dir: Path, chrom: str, *, download: bool) -> Path:
    for name in (f"FG_LD_chr{chrom}.bcor", f"chr{chrom}.bcor", f"{chrom}.bcor"):
        path = ld_bcor_dir / name
        if path.exists():
            return path
    if download:
        normalized = "23" if chrom.upper() == "X" else chrom
        destination = ld_bcor_dir / f"FG_LD_chr{normalized}.bcor"
        url = f"https://storage.googleapis.com/finngen-public-data-ld/imputation_panel_v1/FG_LD_chr{normalized}.bcor"
        logger.warning("Downloading large FinnGen LD BCOR file for chromosome %s to %s", chrom, destination)
        return download_file(url, destination)
    raise FileNotFoundError(f"Could not find BCOR file for chromosome {chrom} in {ld_bcor_dir}")


def _run_ldstore(ldstore_exe: str, bcor: Path, start: int, end: int, table_path: Path) -> None:
    command = [
        ldstore_exe,
        "--bcor",
        str(bcor),
        "--incl-range",
        f"{start}-{end}",
        "--table",
        str(table_path),
    ]
    logger.info("Running LDstore: %s", " ".join(command))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if "Cannot recognize flag '--bcor'" in message or "LDstore v2" in message:
            raise RuntimeError(
                "LDstore rejected the v1.1 BCOR extraction flags used for FinnGen public LD files. "
                "Use LDstore v1.1 with --ld-bcor-dir, or provide precomputed matrices via --ld-matrix-dir."
            ) from subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)


def _ldstore_table_to_matrix(table_path: Path, variants: pl.DataFrame, output_path: Path) -> None:
    positions = variants.get_column("pos").to_list()
    index = {int(pos): idx for idx, pos in enumerate(positions)}
    matrix = np.eye(len(positions), dtype=float)
    if not table_path.exists():
        raise FileNotFoundError(f"LDstore table was not created: {table_path}")

    with table_path.open(newline="") as handle:
        rows = [re.split(r"\s+", line.strip()) for line in handle if line.strip()]
    if not rows:
        _write_ld_matrix(matrix, output_path)
        return

    header = [cell.lower() for cell in rows[0]]
    data_rows = rows[1:] if any(not _is_number(cell) for cell in rows[0]) else rows
    colmap = _ld_table_columns(header) if data_rows is not rows else None
    for row in data_rows:
        try:
            if colmap:
                pos_a = int(float(row[colmap["pos_a"]]))
                pos_b = int(float(row[colmap["pos_b"]]))
                r = float(row[colmap["r"]])
            else:
                pos_a, pos_b, r = _infer_ld_row(row)
        except (ValueError, IndexError, KeyError):
            continue
        if pos_a in index and pos_b in index:
            i = index[pos_a]
            j = index[pos_b]
            matrix[i, j] = r
            matrix[j, i] = r
    _write_ld_matrix(matrix, output_path)


def _ld_table_columns(header: list[str]) -> dict[str, int]:
    def find(candidates: tuple[str, ...]) -> int:
        for idx, value in enumerate(header):
            cleaned = re.sub(r"[^a-z0-9]+", "", value)
            if any(cleaned == candidate or cleaned.endswith(candidate) for candidate in candidates):
                return idx
        raise KeyError(candidates)

    return {
        "pos_a": find(("pos1", "position1", "bp1", "positiona", "bpa")),
        "pos_b": find(("pos2", "position2", "bp2", "positionb", "bpb")),
        "r": find(("corr", "correlation", "r", "r2", "ld")),
    }


def _infer_ld_row(row: list[str]) -> tuple[int, int, float]:
    numbers = [float(value) for value in row if _is_number(value)]
    if len(numbers) < 3:
        raise ValueError(row)
    # LDstore table rows contain two base-pair positions and a correlation; this
    # fallback takes the first two integer-like fields and the last bounded value.
    positions = [int(value) for value in numbers if float(value).is_integer() and value > 1]
    correlations = [value for value in numbers if -1 <= value <= 1]
    if len(positions) < 2 or not correlations:
        raise ValueError(row)
    return positions[0], positions[1], correlations[-1]


def _write_ld_matrix(matrix: np.ndarray, path: Path) -> None:
    np.savetxt(path, matrix, fmt="%.8g")


def _run_susie(config: FineMappingRunConfig, inputs: dict[str, Path], ld_path: Path, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config.rscript_exe,
        str(config.output_dir / "run_susie_region.R"),
        str(inputs["susie_z"]),
        str(ld_path),
        str(out_prefix),
        str(config.max_causal),
    ]
    logger.info("Running SuSiE: %s", " ".join(command))
    subprocess.run(command, check=True)


def _run_finemap(config: FineMappingRunConfig, inputs: dict[str, Path], ld_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    master_path = out_dir / "region.master"
    master_path.write_text(
        "z;ld;snp;config;cred;log;n_samples\n"
        f"{inputs['finemap_z']};{ld_path};{out_dir / 'region.snp'};{out_dir / 'region.config'};"
        f"{out_dir / 'region.cred'};{out_dir / 'region.log'};{config.n_samples}\n",
        encoding="utf-8",
    )
    command = [
        config.finemap_exe,
        "--sss",
        "--in-files",
        str(master_path),
        "--n-causal-snps",
        str(config.max_causal),
        "--prior-snps",
    ]
    logger.info("Running FINEMAP: %s", " ".join(command))
    subprocess.run(command, check=True)


def _write_susie_runner(path: Path) -> None:
    path.write_text(
        """args <- commandArgs(trailingOnly = TRUE)
z_path <- args[[1]]
ld_path <- args[[2]]
out_prefix <- args[[3]]
L <- as.integer(args[[4]])

suppressPackageStartupMessages(library(susieR))

z <- read.delim(z_path, check.names = FALSE)
R <- as.matrix(read.table(ld_path, header = FALSE))
fit <- susie_rss(
  z = z$z,
  R = R,
  prior_weights = z$susie_prior_weight,
  L = L
)
out <- data.frame(
  variant_id = z$variant_id,
  z = z$z,
  prior_weight = z$susie_prior_weight,
  pip = fit$pip
)
write.table(out, paste0(out_prefix, ".pip.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
saveRDS(fit, paste0(out_prefix, ".rds"))
""",
        encoding="utf-8",
    )


def _safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:180] or "region"


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def check_fine_mapping_tools(config: FineMappingRunConfig) -> list[str]:
    """Return missing external tools needed for the requested run."""

    missing: list[str] = []
    if config.ld_bcor_dir and shutil.which(config.ldstore_exe) is None:
        missing.append(config.ldstore_exe)
    elif config.ld_bcor_dir:
        ldstore_problem = _ldstore_cli_problem(config.ldstore_exe)
        if ldstore_problem:
            missing.append(ldstore_problem)
    if config.run_susie and shutil.which(config.rscript_exe) is None:
        missing.append(config.rscript_exe)
    if config.run_finemap and shutil.which(config.finemap_exe) is None:
        missing.append(config.finemap_exe)
    return missing


def _ldstore_cli_problem(ldstore_exe: str) -> str | None:
    try:
        result = subprocess.run(
            [ldstore_exe, "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    help_text = f"{result.stdout}\n{result.stderr}"
    if "LDstore v2" in help_text or "--bcor-to-text" in help_text:
        return f"{ldstore_exe} (LDstore v2 detected; use LDstore v1.1 with --ld-bcor-dir)"
    return None
