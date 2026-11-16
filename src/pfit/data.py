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
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def reference() -> pd.DataFrame:
    """One row per DMS assay: target sequence, selection type, sizes, MSA depth, ..."""
    path = _download(REFERENCE_URL, DATA_DIR / "reference" / "DMS_substitutions.csv")
    return pd.read_csv(path)


def published_spearman() -> pd.DataFrame:
    """Per-assay zero-shot Spearman published on the ProteinGym leaderboard."""
    path = _download(
        PUBLISHED_SPEARMAN_URL, DATA_DIR / "reference" / "DMS_substitutions_Spearman_DMS_level.csv"
    )
    return pd.read_csv(path).rename(columns={"DMS ID": "DMS_id"}).set_index("DMS_id")


@dataclass
class Assay:
    dms_id: str
    target_seq: str
    selection_type: str
    df: pd.DataFrame  # columns: mutant, mutated_sequence, DMS_score, DMS_score_bin


def load_assay(dms_id: str) -> Assay:
    ref = reference().set_index("DMS_id")
    if dms_id not in ref.index:
        raise KeyError(f"unknown DMS_id {dms_id!r}")
    row = ref.loc[dms_id]
    csv_path = DATA_DIR / "DMS_ProteinGym_substitutions" / row["DMS_filename"]
    if not csv_path.exists():
        _extract(row["DMS_filename"])
    df = pd.read_csv(csv_path)
    return Assay(dms_id, row["target_seq"], row["coarse_selection_type"], df)


def _extract(filename: str) -> None:
    zpath = _download(DMS_ZIP_URL, DATA_DIR / "raw" / "DMS_ProteinGym_substitutions.zip")
    out_dir = DATA_DIR / "DMS_ProteinGym_substitutions"
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        matches = [n for n in zf.namelist() if n.endswith("/" + filename) or n == filename]
        if not matches:
            raise FileNotFoundError(f"{filename} not in {zpath.name}")
        (out_dir / filename).write_bytes(zf.read(matches[0]))
