"""Project configuration loaded from YAML (defaults to configs/default.yaml).

The config object is intentionally flat-ish so it can be passed around as a
single dependency. Paths are resolved relative to the project root unless
absolute. Build assumptions (GWAS / entropy / LD reference) are surfaced here
so the liftover step is opt-out via configuration rather than hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class _Base(BaseModel):
    model_config = ConfigDict(validate_assignment=True)


class Paths(_Base):
    project_root: Path = PROJECT_ROOT
    entropy_dir: Path = Path("data/entropy")
    gwas_raw: Path = Path("data/gwas/EUR_Metal_LDSC-CORR_Neff.v2.txt")
    gwas_harmonised: Path = Path("data/gwas/sumstats.hg19.parquet")
    background: Path = Path("data/background/entropy_bg.npz")
    reference_dir: Path = Path("data/reference")
    ld_cache: Path = Path("data/ld_cache")
    polyfun_dir: Path = Path("external/polyfun")
    finemap_exe: Path = Path("external/bin/finemap")
    output_dir: Path = Path("output")

    def absolute(self, attr: str) -> Path:
        p: Path = getattr(self, attr)
        return p if p.is_absolute() else (self.project_root / p).resolve()


class Builds(_Base):
    """Reference genome build assumptions for the three inputs."""

    gwas: str = "hg19"
    entropy: str = "hg38"
    ld: str = "hg19"


class Background(_Base):
    n_samples: int = 10_000_000
    n_bins: int = 500
    entropy_min: float = 0.0
    entropy_max: float = 3.0
    seed: int = 42


class PriorParams(_Base):
    tau: float = Field(1.0, description="Temperature on the surprise score.")
    epsilon: float = Field(
        1e-6, description="Density floor to avoid log(0) on tail values."
    )


LdMode = Literal["precomputed_npz", "plink"]


class FineMapParams(_Base):
    """LD source for FINEMAP (PolyFun wrapper).

    * ``precomputed_npz`` — download Broad-style per-region NPZ matrices; use
      :attr:`ld_npz_url_prefix` for the base URL and optional
      :attr:`ld_regions_file` for genome-wide tiling + full ``--ld`` URLs.
    * ``plink`` — compute LD from a Plink genotype prefix (``--geno`` path
      without ``.bed``); use ancestry-matched reference panels when GWAS is not
      UKB-like EUR.
    """

    method: str = "finemap"
    max_num_causal: int = 5
    memory_gb: int = 4
    threads: int = 4
    locus_window_bp: int = 1_500_000
    ld_mode: LdMode = "precomputed_npz"
    # Base URL for demo / per-block ``--ld`` when using UKB tiling (slash-terminated).
    # YAML may still use the legacy key ``ukb_ld_url_prefix``.
    ld_npz_url_prefix: str = Field(
        default="https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/",
        validation_alias=AliasChoices("ld_npz_url_prefix", "ukb_ld_url_prefix"),
    )
    # Optional PolyFun-style regions file (CHR, START, END, URL_PREFIX per row).
    # Default: PolyFun’s bundled ukb_regions.tsv.gz when unset.
    ld_regions_file: Path | None = None
    # Plink prefix (no .bed/.bim/.fam suffix) when ld_mode == "plink".
    ld_plink_prefix: Path | None = None
    pvalue_cutoff: float = 5e-8
    max_concurrent_jobs: int = 4


class GwasDataset(_Base):
    """Labels this GWAS / ancestry run (for logs and optional path layout).

    When you pass ``--gwas-id SLUG`` on the CLI (and SLUG is not ``default``),
    :func:`apply_gwas_cli_overrides` routes harmonised sumstats and pipeline
    outputs under ``data/gwas/{SLUG}/`` and ``output/{SLUG}/``. You can
    instead set ``paths.*`` explicitly in YAML for full control.

    Set ``auto_paths: true`` with a non-default ``id`` to get the same layout
    from YAML alone (no ``--gwas-id`` on the command line).
    """

    id: str = "default"
    auto_paths: bool = False


class Config(_Base):
    paths: Paths = Paths()
    builds: Builds = Builds()
    background: Background = Background()
    prior: PriorParams = PriorParams()
    finemap: FineMapParams = FineMapParams()
    gwas_dataset: GwasDataset = GwasDataset()


def load_config(path: Path | None = None) -> Config:
    """Load YAML config; missing fields fall back to model defaults."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return Config()
    with cfg_path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


def apply_gwas_cli_overrides(
    cfg: Config,
    *,
    gwas_id: str | None = None,
    gwas_raw: Path | str | None = None,
) -> None:
    """Apply ``--gwas-id`` / ``--gwas-raw`` from the CLI (mutates ``cfg``).

    * ``--gwas-raw PATH`` — overrides ``paths.gwas_raw`` whenever provided.
    * ``--gwas-id SLUG`` — when this argument is **passed** (not omitted), sets
      ``gwas_dataset.id`` to ``SLUG`` and, if ``SLUG`` is not ``default``,
      convention paths: ``output/{SLUG}/`` and
      ``data/gwas/{SLUG}/sumstats.hg19.parquet`` for harmonised output.

    If you omit ``--gwas-id`` entirely, ``gwas_dataset`` and ``paths`` from YAML
    are left unchanged (use a per-study YAML for bespoke paths).

    For layout driven purely from YAML without repeating paths, set
    ``gwas_dataset.auto_paths: true`` and a non-default ``gwas_dataset.id``.
    """
    if gwas_raw is not None:
        cfg.paths.gwas_raw = Path(gwas_raw)

    if gwas_id is not None:
        slug = gwas_id.strip() or "default"
        cfg.gwas_dataset.id = slug
        if slug != "default":
            cfg.paths.output_dir = Path("output") / slug
            cfg.paths.gwas_harmonised = Path("data/gwas") / slug / "sumstats.hg19.parquet"
            cfg.paths.ld_cache = Path("data/ld_cache") / slug


def apply_gwas_auto_paths_from_yaml(cfg: Config) -> None:
    """If ``gwas_dataset.auto_paths`` is true, mirror :func:`apply_gwas_cli_overrides` layout."""
    if not cfg.gwas_dataset.auto_paths:
        return
    slug = (cfg.gwas_dataset.id or "default").strip() or "default"
    if slug == "default":
        return
    cfg.paths.output_dir = Path("output") / slug
    cfg.paths.gwas_harmonised = Path("data/gwas") / slug / "sumstats.hg19.parquet"
    cfg.paths.ld_cache = Path("data/ld_cache") / slug


def resolve_project_path(cfg: Config, p: Path | None) -> Path | None:
    """Resolve a path relative to ``paths.project_root`` if not absolute."""
    if p is None:
        return None
    return p if p.is_absolute() else (cfg.paths.project_root / p).resolve()


def resolve_plink_prefix(cfg: Config) -> Path:
    """Absolute path **stem** ``P`` (no ``.bed`` suffix) with ``P.bed/.bim/.fam`` on disk."""
    raw = cfg.finemap.ld_plink_prefix
    if raw is None:
        raise ValueError(
            "finemap.ld_mode='plink' requires finemap.ld_plink_prefix "
            "(Plink prefix without the .bed extension)."
        )
    p = resolve_project_path(cfg, Path(raw))
    if p is None:
        raise ValueError("Could not resolve finemap.ld_plink_prefix.")
    s = str(p)
    base = Path(s[:-4]) if s.endswith(".bed") else p
    missing: list[str] = []
    for ext in (".bed", ".bim", ".fam"):
        part = Path(str(base) + ext)
        if not part.exists():
            missing.append(str(part))
    if missing:
        raise FileNotFoundError(
            "Plink LD genotype triplet incomplete for prefix "
            f"{base}: missing {missing}"
        )
    return base.resolve()


def validate_finemap_ld(cfg: Config) -> None:
    """Fail fast on inconsistent LD settings before spawning FINEMAP."""
    if cfg.finemap.ld_mode == "plink":
        resolve_plink_prefix(cfg)
        return
    if cfg.finemap.ld_regions_file is not None:
        rf = resolve_project_path(cfg, Path(cfg.finemap.ld_regions_file))
        if rf is None or not rf.exists():
            raise FileNotFoundError(
                f"finemap.ld_regions_file not found: {cfg.finemap.ld_regions_file}"
            )


def apply_ld_cli_overrides(
    cfg: Config,
    *,
    ld_mode: str | None = None,
    ld_npz_url_prefix: str | None = None,
    ld_plink_prefix: Path | str | None = None,
    ld_regions_file: Path | str | None = None,
) -> None:
    """Optional CLI overrides for LD (mutates ``cfg``)."""
    if ld_mode is not None:
        m = ld_mode.strip().lower().replace("-", "_")
        if m not in ("precomputed_npz", "plink"):
            raise ValueError(
                f"Invalid --ld-mode {ld_mode!r}; use precomputed_npz or plink."
            )
        cfg.finemap.ld_mode = m  # type: ignore[assignment]
    if ld_npz_url_prefix is not None:
        cfg.finemap.ld_npz_url_prefix = ld_npz_url_prefix
    if ld_plink_prefix is not None:
        cfg.finemap.ld_plink_prefix = Path(ld_plink_prefix)
    if ld_regions_file is not None:
        cfg.finemap.ld_regions_file = Path(ld_regions_file)


def load_pipeline_config(
    path: Path | None = None,
    *,
    gwas_id: str | None = None,
    gwas_raw: Path | str | None = None,
    ld_mode: str | None = None,
    ld_npz_url_prefix: str | None = None,
    ld_plink_prefix: Path | str | None = None,
    ld_regions_file: Path | str | None = None,
) -> Config:
    """Load YAML, optional YAML-driven auto paths, then CLI GWAS + LD overrides."""
    cfg = load_config(path)
    apply_gwas_auto_paths_from_yaml(cfg)
    apply_gwas_cli_overrides(cfg, gwas_id=gwas_id, gwas_raw=gwas_raw)
    apply_ld_cli_overrides(
        cfg,
        ld_mode=ld_mode,
        ld_npz_url_prefix=ld_npz_url_prefix,
        ld_plink_prefix=ld_plink_prefix,
        ld_regions_file=ld_regions_file,
    )
    return cfg
