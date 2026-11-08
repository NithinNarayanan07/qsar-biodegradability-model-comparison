"""Acquire the original UCI QSAR data and preserve its source provenance.

The unmodified, headerless source is kept in ``data/biodeg.csv``. Existing local
data is verified and reused, allowing subsequent training runs to work offline.
A pinned SHA-256 checksum identifies the exact source version used to produce
the project's model results; source URLs, attribution, and acquisition details
are saved separately in ``data/source_metadata.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import zipfile

DATASET_URL = "https://archive.ics.uci.edu/dataset/254/qsar+biodegradation"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/254/qsar+biodegradation.zip"
# Pin the CSV content, rather than ZIP packaging details that can change even
# when the underlying data is identical.
RAW_SHA256 = "c66ac1a23aba11fb166fd5054fea14ac97b8e4f7e4940ae679dc863d1cc4b3c9"
CITATION = (
    "Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. "
    "(2013). QSAR biodegradation [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5H60M"
)


def sha256_file(path: Path) -> str:
    """Identify a file by its exact bytes, independent of its name or timestamp."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_dataset(data_dir: Path) -> Path:
    """Return a verified source CSV, downloading it only when it is absent.

    Preserve the raw bytes and create provenance metadata on first acquisition.
    Raise ValueError when local or downloaded bytes differ from the known source
    checksum, so a changed dataset cannot silently replace the reviewed version.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "biodeg.csv"
    metadata_path = data_dir / "source_metadata.json"
    archive_hash = None
    acquisition = "verified existing local file"

    # Reuse the checked-in raw data for reproducible, offline lab execution.
    if raw_path.exists():
        raw = raw_path.read_bytes()
    else:
        request = urllib.request.Request(
            DOWNLOAD_URL, headers={"User-Agent": "QSAR-learning-project/1.0"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read()
        archive_hash = hashlib.sha256(archive).hexdigest()
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            # Read only the expected member without extracting archive paths.
            raw = bundle.read("biodeg.csv")
        acquisition = "downloaded from official UCI archive"

    # Verify before writing a downloaded file or using an existing one. Parsing,
    # schema validation, and duplicate auditing are handled by the training step.
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash != RAW_SHA256:
        raise ValueError(
            "The raw data checksum differs from the verified UCI version. "
            "Review the source change before replacing any training artifacts."
        )
    if not raw_path.exists():
        raw_path.write_bytes(raw)

    # Retain the original acquisition record on subsequent runs. The raw CSV is
    # still checksum-verified every time, even when metadata already exists.
    if not metadata_path.exists():
        metadata = {
            "dataset": "QSAR biodegradation",
            "uci_id": 254,
            "dataset_url": DATASET_URL,
            "download_url": DOWNLOAD_URL,
            "doi": "https://doi.org/10.24432/C5H60M",
            "license": "CC BY 4.0",
            "citation": CITATION,
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "acquisition": acquisition,
            "raw_file": "biodeg.csv",
            "raw_sha256": raw_hash,
            "raw_bytes": len(raw),
            "archive_sha256": archive_hash,
            "format": (
                "UTF-8; semicolon separated; no header; " "41 features followed by NRB/RB class"
            ),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Verified original data: {raw_path} (SHA-256 {raw_hash})")
    return raw_path


def main() -> None:
    """Fetch or verify the dataset in the requested local data directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    arguments = parser.parse_args()
    fetch_dataset(arguments.data_dir)


if __name__ == "__main__":
    main()
