"""Utilities for the chromosome-level GPN-Star entropy parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

EXPECTED_COLUMNS = ("chrom", "pos", "ref", "entropy_calibrated")
CHROMOSOMES = tuple(str(chrom) for chrom in range(1, 23)) + ("X", "Y")


@dataclass(frozen=True)
class EntropyFileInfo:
    """Metadata for one chromosome entropy parquet file."""

    chrom: str
    path: Path
    rows: int | None
    columns: tuple[str, ...]
    size_bytes: int

    @property
    def missing_columns(self) -> tuple[str, ...]:
        return tuple(column for column in EXPECTED_COLUMNS if column not in self.columns)


def normalize_chrom(chrom: str | int) -> str:
    """Return FinnGen-compatible chromosome names without a leading ``chr``."""

    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    if value == "23":
        return "X"
    if value == "24":
        return "Y"
    return value.upper()


def entropy_path(entropy_dir: Path, chrom: str | int) -> Path:
    """Return the expected entropy parquet path for a chromosome."""

    return entropy_dir / f"entropy_chr{normalize_chrom(chrom)}.parquet"


def list_entropy_files(entropy_dir: Path) -> list[Path]:
    """List chromosome entropy parquet files in deterministic chromosome order."""

    files = [entropy_path(entropy_dir, chrom) for chrom in CHROMOSOMES]
    present = [path for path in files if path.exists()]
    extras = sorted(entropy_dir.glob("entropy_chr*.parquet"))
    seen = set(present)
    return present + [path for path in extras if path not in seen]


def scan_entropy_chrom(entropy_dir: Path, chrom: str | int) -> pl.LazyFrame:
    """Scan one chromosome entropy parquet file lazily."""

    path = entropy_path(entropy_dir, chrom)
    if not path.exists():
        raise FileNotFoundError(f"Missing entropy parquet for chromosome {chrom}: {path}")

    return (
        pl.scan_parquet(path)
        .select(EXPECTED_COLUMNS)
        .with_columns(
            pl.col("chrom").cast(pl.Utf8).map_elements(normalize_chrom, return_dtype=pl.Utf8),
            pl.col("ref").cast(pl.Utf8).str.to_uppercase(),
            pl.col("pos").cast(pl.Int64),
            pl.col("entropy_calibrated").cast(pl.Float64),
        )
    )


def inspect_entropy_files(entropy_dir: Path) -> list[EntropyFileInfo]:
    """Collect cheap metadata for all visible entropy parquet files."""

    if not entropy_dir.exists():
        raise FileNotFoundError(f"Entropy directory does not exist: {entropy_dir}")

    infos: list[EntropyFileInfo] = []
    for path in list_entropy_files(entropy_dir):
        schema = pl.scan_parquet(path).collect_schema()
        chrom = path.stem.removeprefix("entropy_chr")
        rows = _parquet_row_count(path)
        infos.append(
            EntropyFileInfo(
                chrom=normalize_chrom(chrom),
                path=path,
                rows=rows,
                columns=tuple(schema.names()),
                size_bytes=path.stat().st_size,
            )
        )
    return infos


def validate_entropy_files(entropy_dir: Path) -> list[str]:
    """Return human-readable validation problems for the entropy dataset."""

    problems: list[str] = []
    infos = inspect_entropy_files(entropy_dir)
    found = {info.chrom for info in infos}
    for chrom in CHROMOSOMES:
        if chrom not in found:
            problems.append(f"Missing entropy_chr{chrom}.parquet")

    for info in infos:
        for column in info.missing_columns:
            problems.append(f"{info.path.name} is missing column {column!r}")
    return problems


def _parquet_row_count(path: Path) -> int | None:
    """Read row count from parquet metadata when pyarrow is available."""

    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return None

    return pq.ParquetFile(path).metadata.num_rows
