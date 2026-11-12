"""The score: instrument hours to collect 75% of a session's curated particles.

Summed over the sessions played, and read against three references:

* vs operator   hours relative to the session's recorded acquisition order
* vs EPU raster hours relative to EPU's own cluster-by-cluster raster (blind to quality)
* gap closed    (operator - you) / (operator - oracle): 0% is the operator, 100% the
                oracle that knows every outcome. Negative means slower than the operator.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from .data import DATA_DIR
from .references import REFERENCES

REFERENCES_FILE = DATA_DIR / "references.json"


def load_references(path=REFERENCES_FILE) -> dict:
    """{session: {reference: hours}} for the training sessions (python main.py references)."""
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def scorecard(entries: Mapping[str, Mapping[str, float]], references: Mapping[str, Mapping[str, float]]) -> pd.DataFrame:
    """entries: {name: {session: hours}}; references: {session: {reference: hours}}.

    One row per entry and reference over the sessions every entry played."""
    sessions = [s for s in references if all(s in e for e in entries.values())]
    rows = {}
    for ref in REFERENCES:
        rows[ref] = {s: references[s][ref] for s in sessions}
    rows.update({name: {s: e[s] for s in sessions} for name, e in entries.items()})
    df = pd.DataFrame(rows).T[sessions]
    df["total h"] = df[sessions].sum(axis=1)
    op, orc, ras = df.loc["operator", "total h"], df.loc["oracle", "total h"], df.loc["epu_raster", "total h"]
    df["vs operator"] = 100.0 * (df["total h"] / op - 1.0)
    df["vs EPU raster"] = 100.0 * (df["total h"] / ras - 1.0)
    df["gap closed"] = 100.0 * (op - df["total h"]) / (op - orc)
    return df.sort_values("total h")


def format_scorecard(df: pd.DataFrame) -> str:
    sessions = [c for c in df.columns if c not in ("total h", "vs operator", "vs EPU raster", "gap closed")]
    head = "| entry | " + " | ".join(sessions) + " | total | vs operator | vs EPU raster | gap closed |"
    sep = "|---" * (len(sessions) + 5) + "|"
    lines = [head, sep]
    for name, r in df.iterrows():
        cells = [f"{r[s]:.2f} h" for s in sessions]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['total h']:.2f} h | {r['vs operator']:+.1f}% | "
                     f"{r['vs EPU raster']:+.1f}% | {r['gap closed']:.0f}% |")
    return "\n".join(lines)
