"""Project configuration loaded from YAML.

The config object is intentionally flat-ish so it can be passed around as a
single dependency. Paths are resolved relative to the project root unless
absolute. Build assumptions (GWAS / entropy / LD reference) are surfaced here
so the liftover step is opt-out via configuration rather than hard-coded.

Every CLI flag has a YAML counterpart so the full pipeline can be driven
from one ``-c configs/your.yaml``. CLI flags, when passed, override YAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

PriorMode = Literal["none", "entropy", "entropy_raw"]


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
    loci: Path = Path("configs/loci_demo.tsv")

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
    mode: PriorMode = Field(
        "entropy",
        description=(
            "Prior mode: 'none' (uniform / non-functional), 'entropy' "
            "(surprise vs. genome-wide background), or 'entropy_raw' "
            "(negated raw per-SNP entropy, no background)."
        ),
    )
    tau: float = Field(
        1.0,
        description=(
            "Temperature on the per-SNP score; for 'entropy' it scales "
            "-log f_bg(e), for 'entropy_raw' it scales -e directly."
        ),
    )
    epsilon: float = Field(
        1e-6, description="Density floor to avoid log(0) on tail values."
    )


LdMode = Literal["precomputed_npz", "plink"]


class FineMapParams(_Base):
    """LD source for FINEMAP (PolyFun wrapper).

    * ``precomputed_npz`` — download Broad-style per-region NPZ matrices; use
      :attr:`ld_npz_url_prefix` for the base URL.
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
    ld_npz_url_prefix: str = Field(
        default="https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/",
        validation_alias=AliasChoices("ld_npz_url_prefix", "ukb_ld_url_prefix"),
    )
    ld_plink_prefix: Path | None = None
    max_concurrent_jobs: int = 4


class Config(_Base):
    paths: Paths = Paths()
    builds: Builds = Builds()
    background: Background = Background()
    prior: PriorParams = PriorParams()
    finemap: FineMapParams = FineMapParams()


def load_config(path: Path | None = None) -> Config:
    """Load YAML config; missing fields fall back to model defaults."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return Config()
    with cfg_path.open("r") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)


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


def apply_cli_overrides(
    cfg: Config,
    *,
    gwas_raw: Path | str | None = None,
    output_dir: Path | str | None = None,
    loci: Path | str | None = None,
    prior_mode: str | None = None,
    ld_mode: str | None = None,
    ld_npz_url_prefix: str | None = None,
    ld_plink_prefix: Path | str | None = None,
) -> None:
    """Apply CLI overrides to ``cfg`` in place.

    Each kwarg mirrors one CLI flag; when not ``None`` it wins over YAML.
    """
    if gwas_raw is not None:
        cfg.paths.gwas_raw = Path(gwas_raw)
    if output_dir is not None:
        cfg.paths.output_dir = Path(output_dir)
    if loci is not None:
        cfg.paths.loci = Path(loci)
    if prior_mode is not None:
        m = prior_mode.strip().lower()
        if m not in ("none", "entropy", "entropy_raw"):
            raise ValueError(
                f"Invalid --prior {prior_mode!r}; use none, entropy, or entropy_raw."
            )
        cfg.prior.mode = m  # type: ignore[assignment]
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


def load_pipeline_config(
    path: Path | None = None,
    *,
    gwas_raw: Path | str | None = None,
    output_dir: Path | str | None = None,
    loci: Path | str | None = None,
    prior_mode: str | None = None,
    ld_mode: str | None = None,
    ld_npz_url_prefix: str | None = None,
    ld_plink_prefix: Path | str | None = None,
) -> Config:
    """Load YAML, then apply CLI overrides."""
    cfg = load_config(path)
    apply_cli_overrides(
        cfg,
        gwas_raw=gwas_raw,
        output_dir=output_dir,
        loci=loci,
        prior_mode=prior_mode,
        ld_mode=ld_mode,
        ld_npz_url_prefix=ld_npz_url_prefix,
        ld_plink_prefix=ld_plink_prefix,
    )
    return cfg
