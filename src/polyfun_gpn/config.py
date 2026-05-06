"""Project configuration loaded from YAML (defaults to configs/default.yaml).

The config object is intentionally flat-ish so it can be passed around as a
single dependency. Paths are resolved relative to the project root unless
absolute. Build assumptions (GWAS / entropy / LD reference) are surfaced here
so the liftover step is opt-out via configuration rather than hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


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


class FineMapParams(_Base):
    method: str = "finemap"
    max_num_causal: int = 5
    memory_gb: int = 4
    threads: int = 4
    locus_window_bp: int = 1_500_000
    ukb_ld_url_prefix: str = (
        "https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/"
    )
    pvalue_cutoff: float = 5e-8
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
