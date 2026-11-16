"""Write daily summaries with stable column alignment."""

from pathlib import Path

import pandas as pd


def case_slug(name):
    """Convert public case labels to portable directory names."""
    return name.replace(" ", "-")


def append_summary(row, save_path):
    """Union iteration columns so runs of different lengths do not corrupt the CSV.

    Summary tables contain only one row per day. Rewriting this small table also
    preserves missing fields when the first day fails before refinement starts.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        columns = list(existing.columns)
        columns.extend(c for c in incoming.columns if c not in columns)
        incoming = pd.concat(
            [existing.reindex(columns=columns), incoming.reindex(columns=columns)],
            ignore_index=True,
        )
    incoming.to_csv(path, index=False)
