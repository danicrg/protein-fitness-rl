"""ProteinGym DMS substitution data: download, cache, load.

Data lives in ``$PFIT_DATA`` (default ``<repo>/data``), which is git-ignored.
Source: https://proteingym.org (Notin et al. 2023), version pinned below.
"""

from __future__ import annotations

import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROTEINGYM_VERSION = "v1.3"
_GH = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main"
REFERENCE_URL = f"{_GH}/reference_files/DMS_substitutions.csv"
PUBLISHED_SPEARMAN_URL = (
    f"{_GH}/benchmarks/DMS_zero_shot/substitutions/Spearman/"
    "DMS_substitutions_Spearman_DMS_level.csv"
)
DMS_ZIP_URL = (
    f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{PROTEINGYM_VERSION}/"
    "DMS_ProteinGym_substitutions.zip"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PFIT_DATA", REPO_ROOT / "data"))

# Small, single-mutant assays spanning the five ProteinGym selection types.
# VKOR1 appears twice on purpose: same protein, two phenotypes (activity vs abundance).
STARTER_ASSAYS = [
    "B2L11_HUMAN_Dutta_2010_binding-Mcl-1",  # Binding
    "VKOR1_HUMAN_Chiasson_2020_activity",  # Activity
    "ENVZ_ECOLI_Ghose_2023",  # Activity
    "VKOR1_HUMAN_Chiasson_2020_abundance",  # Expression
    "NUD15_HUMAN_Suiter_2020",  # Expression
    "OTU7A_HUMAN_Tsuboyama_2023_2L2D",  # Stability
    "SQSTM_MOUSE_Tsuboyama_2023_2RRU",  # Stability
    "BLAT_ECOLX_Jacquier_2013",  # OrganismalFitness
    "RL40A_YEAST_Roscoe_2013",  # OrganismalFitness
]


def _download(url: str, dest: Path) -> Path:
    """Given plumbing: fetch url to dest once, return dest."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def reference() -> pd.DataFrame:
    """One row per DMS assay (217): DMS_id, DMS_filename, target_seq, coarse_selection_type, ...

    Use _download(REFERENCE_URL, <path under DATA_DIR>) to fetch and cache it.
    """
    raise NotImplementedError("stage 3")


def published_spearman() -> pd.DataFrame:
    """The leaderboard: one row per assay, one column per model (e.g. "ESM2 (8M)").

    The raw CSV names its id column "DMS ID"; return it indexed by a column named "DMS_id".
    """
    raise NotImplementedError("stage 3")


@dataclass
class Assay:
    dms_id: str
    target_seq: str  # the wild-type sequence
    selection_type: str  # Binding, Activity, Expression, Stability or OrganismalFitness
    df: pd.DataFrame  # columns: mutant, mutated_sequence, DMS_score, DMS_score_bin


def load_assay(dms_id: str) -> Assay:
    """Look the assay up in reference(), make sure its CSV is extracted (see _extract), load it.

    Raise KeyError for an unknown dms_id.
    """
    raise NotImplementedError("stage 3")


def _extract(filename: str) -> None:
    """Given plumbing: pull one assay CSV out of the ProteinGym zip into DATA_DIR."""
    zpath = _download(DMS_ZIP_URL, DATA_DIR / "raw" / "DMS_ProteinGym_substitutions.zip")
    out_dir = DATA_DIR / "DMS_ProteinGym_substitutions"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        matches = [n for n in zf.namelist() if n.endswith("/" + filename) or n == filename]
        if not matches:
            raise FileNotFoundError(f"{filename} not in {zpath.name}")
        (out_dir / filename).write_bytes(zf.read(matches[0]))
