"""Skip Hugging Face's network check when the test model is already cached.

from_pretrained asks the Hub for updates on every call (about 30 s across the suite).
Once facebook/esm2_t6_8M_UR50D is downloaded, run fully offline. The variable must be set
before huggingface_hub is imported, so the cache is located by hand here.
"""

import os
from pathlib import Path

_hub = Path(
    os.environ.get("HF_HUB_CACHE")
    or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
)
if (_hub / "models--facebook--esm2_t6_8M_UR50D" / "snapshots").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
