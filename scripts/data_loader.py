"""Data loading utilities for ESA river discharge CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def list_station_files(csv_dir: Path) -> List[Path]:
    """List station CSV files from the dataset folder.

    Parameters
    ----------
    csv_dir : Path
        Path to the station CSV directory.

    Returns
    -------
    list of Path
        Sorted list of station CSV file paths.

    Notes
    -----
    author: P24EGCP9020
    usage: files = list_station_files(csv_dir)
    use: Discover available stations for experiment selection.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    return sorted(csv_dir.glob("*_Q_Day.Cmd.csv"))


def parse_station_metadata(csv_file: Path) -> Dict[str, str]:
    """Parse metadata key-value fields from a station CSV header.

    Parameters
    ----------
    csv_file : Path
        Path to one station CSV file.

    Returns
    -------
    dict of str to str
        Metadata dictionary extracted from comment lines.

    Notes
    -----
    author: P24EGCP9020
    usage: metadata = parse_station_metadata(csv_file)
    use: Include station context and citation details in analysis outputs.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    metadata: Dict[str, str] = {}
    with csv_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            cleaned = line.lstrip("#").strip()
            if ":" not in cleaned:
                continue
            key, value = cleaned.split(":", maxsplit=1)
            metadata[key.strip().lower().replace(" ", "_")] = value.strip()
    return metadata


def load_station_data(csv_file: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load one station CSV into a clean time-indexed DataFrame.

    Parameters
    ----------
    csv_file : Path
        Path to one station CSV file.

    Returns
    -------
    tuple
        First item is a DataFrame with columns ``discharge``, ``uncertainty`` and
        ``satellite`` indexed by ``datetime``. Second item is station metadata.

    Raises
    ------
    ValueError
        If the table header line cannot be found.

    Notes
    -----
    author: P24EGCP9020
    usage: frame, meta = load_station_data(csv_file)
    use: Standardized input to preprocessing and modeling scripts.
    project: COEN807 Machine Learning for Real-World Data Analytics.
    """
    metadata = parse_station_metadata(csv_file)
    lines = csv_file.read_text(encoding="utf-8").splitlines()

    header_index = None
    for index, line in enumerate(lines):
        if line.strip().startswith("YYYY-MM-DD;"):
            header_index = index
            break

    if header_index is None:
        raise ValueError(f"Could not locate data header in {csv_file}")

    data = pd.read_csv(
        csv_file,
        sep=";",
        skiprows=header_index,
        engine="python",
        skipinitialspace=True,
    )
    data.columns = [col.strip().lower() for col in data.columns]

    data["datetime"] = pd.to_datetime(
        data["yyyy-mm-dd"].str.strip() + " " + data["hh:mm:ss"].str.strip(),
        format="%Y-%m-%d %H:%M:%S",
    )
    data["discharge"] = pd.to_numeric(data["value"], errors="coerce")
    data["uncertainty"] = pd.to_numeric(data["uncertainty"], errors="coerce")
    data["satellite"] = data["satellite"].astype(str).str.strip()

    cleaned = (
        data[["datetime", "discharge", "uncertainty", "satellite"]]
        .dropna(subset=["discharge"])
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"])
        .set_index("datetime")
    )
    return cleaned, metadata
