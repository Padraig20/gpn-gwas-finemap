"""Download and read FinnGen summary-statistic and fine-mapping files."""

from __future__ import annotations

import csv
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl

DEFAULT_RELEASE = 12
DEFAULT_ENDPOINTS = ("T2D", "E4_DM2")


@dataclass(frozen=True)
class FinnGenPaths:
    """Local cached paths for one FinnGen endpoint."""

    endpoint: str
    summary_stats: Path | None
    susie_snp: Path | None
    finemap_snp: Path | None


def release_base_url(release: int) -> str:
    return f"https://storage.googleapis.com/finngen-public-data-r{release}"


def summary_manifest_url(release: int) -> str:
    return f"{release_base_url(release)}/summary_stats/finngen_R{release}_manifest.tsv"


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    """Download a URL atomically unless the destination already exists."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    request = urllib.request.Request(url, headers={"User-Agent": "gpn-finemap/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with NamedTemporaryFile("wb", delete=False, dir=destination.parent) as handle:
                shutil.copyfileobj(response, handle)
                temp_path = Path(handle.name)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    temp_path.replace(destination)
    return destination


def download_endpoint_files(
    cache_dir: Path,
    release: int = DEFAULT_RELEASE,
    endpoint: str = "T2D",
    summary_url: str | None = None,
    susie_snp_url: str | None = None,
    finemap_snp_url: str | None = None,
    overwrite: bool = False,
) -> FinnGenPaths:
    """Download/cache the FinnGen files needed for the lightweight benchmark.

    Summary statistics are resolved from the public release manifest. Fine-mapping
    URLs are not consistently documented across releases, so this tries common
    public GCS paths and also supports explicit URL overrides.
    """

    endpoint_cache = cache_dir / f"finngen_R{release}" / endpoint
    summary = _download_summary_stats(endpoint_cache, release, endpoint, summary_url, overwrite)
    susie = _download_first_available(
        endpoint_cache / f"{endpoint}.SUSIE.snp.bgz",
        [susie_snp_url] if susie_snp_url else fine_mapping_candidate_urls(release, endpoint, "SUSIE", "snp"),
        overwrite=overwrite,
        required=False,
    )
    finemap = _download_first_available(
        endpoint_cache / f"{endpoint}.FINEMAP.snp.bgz",
        [finemap_snp_url] if finemap_snp_url else fine_mapping_candidate_urls(release, endpoint, "FINEMAP", "snp"),
        overwrite=overwrite,
        required=False,
    )
    return FinnGenPaths(endpoint=endpoint, summary_stats=summary, susie_snp=susie, finemap_snp=finemap)


def fine_mapping_candidate_urls(release: int, endpoint: str, method: str, suffix: str) -> list[str]:
    """Return plausible public FinnGen fine-mapping URLs for a release."""

    base = release_base_url(release)
    method_lower = method.lower()
    filename = f"{endpoint}.{method}.{suffix}.bgz"
    release_filename = f"finngen_R{release}_{filename}"
    return [
        f"{base}/finemapping/{filename}",
        f"{base}/finemapping/{release_filename}",
        f"{base}/finemapping/{method_lower}/{filename}",
        f"{base}/finemapping/{method_lower}/{release_filename}",
        f"{base}/finemapping_results/{filename}",
        f"{base}/finemapping_results/{release_filename}",
    ]


def scan_summary_stats(path: Path) -> pl.LazyFrame:
    """Scan FinnGen summary stats and standardize core columns."""

    return (
        pl.scan_csv(path, separator="\t", infer_schema_length=10_000, null_values=["NA", "nan"])
        .rename({"#chrom": "chrom"})
        .with_columns(
            pl.col("chrom").cast(pl.Utf8).str.replace("^chr", "").str.to_uppercase(),
            pl.col("pos").cast(pl.Int64),
            pl.col("ref").cast(pl.Utf8).str.to_uppercase(),
            pl.col("alt").cast(pl.Utf8).str.to_uppercase(),
            pl.col("pval").cast(pl.Float64),
            pl.col("beta").cast(pl.Float64),
            pl.col("sebeta").cast(pl.Float64),
        )
    )


def scan_finemap_snps(path: Path, method: str) -> pl.LazyFrame:
    """Scan a FinnGen SuSiE or FINEMAP SNP-level fine-mapping file."""

    return (
        pl.scan_csv(path, separator="\t", infer_schema_length=10_000, null_values=["NA", "nan"])
        .rename({"chromosome": "chrom", "position": "pos", "allele1": "ref", "allele2": "alt"})
        .with_columns(
            pl.lit(method.upper()).alias("method"),
            pl.col("chrom").cast(pl.Utf8).str.replace("^chr", "").str.to_uppercase(),
            pl.col("pos").cast(pl.Int64),
            pl.col("ref").cast(pl.Utf8).str.to_uppercase(),
            pl.col("alt").cast(pl.Utf8).str.to_uppercase(),
            pl.col("prob").cast(pl.Float64).alias("pip"),
        )
    )


def _download_summary_stats(
    endpoint_cache: Path,
    release: int,
    endpoint: str,
    explicit_url: str | None,
    overwrite: bool,
) -> Path | None:
    destination = endpoint_cache / f"{endpoint}.gz"
    if explicit_url:
        return download_file(explicit_url, destination, overwrite=overwrite)

    manifest_path = endpoint_cache.parent / "finngen_manifest.tsv"
    download_file(summary_manifest_url(release), manifest_path, overwrite=overwrite)
    url = resolve_summary_url_from_manifest(manifest_path, endpoint)
    if url is None:
        return None
    return download_file(url, destination, overwrite=overwrite)


def resolve_summary_url_from_manifest(manifest_path: Path, endpoint: str) -> str | None:
    """Find the summary-statistic URL for an endpoint in a FinnGen manifest."""

    endpoint = endpoint.upper()
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            values = {key: (value or "") for key, value in row.items()}
            haystack = "\t".join(values.values()).upper()
            if f"/{endpoint}.GZ" not in haystack and f"\t{endpoint}\t" not in f"\t{haystack}\t":
                continue
            for value in values.values():
                if value.startswith("http") and value.endswith(f"/{endpoint}.gz"):
                    return value
                if value.startswith("http") and value.endswith(f"{endpoint}.gz"):
                    return value
    return None


def _download_first_available(
    destination: Path,
    urls: list[str],
    overwrite: bool,
    required: bool,
) -> Path | None:
    errors: list[str] = []
    for url in urls:
        if not url:
            continue
        try:
            return download_file(url, destination, overwrite=overwrite)
        except RuntimeError as exc:
            errors.append(str(exc))

    if required:
        message = "\n".join(errors[-3:])
        raise RuntimeError(f"None of the candidate URLs worked for {destination.name}:\n{message}")
    return None
