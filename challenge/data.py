"""The data: one folder per recorded session.

    data/<session>/
      session.json        name, specimen, optics, counts, the session's own instrument clock
      squares.parquet     one row per grid square (positions on the grid, hole-finder summaries)
      holes.parquet       one row per hole the operator selected (position, size, hole-finder features)
      micrographs.parquet one row per exposure: CryoSPARC Live outputs and the two labels
      embeddings.npy      768-d Cryo-IEF vector of every hole's crop, rows aligned with holes.parquet
      hole_images.npy     64x64 crop of the square overview around every hole, rows aligned with holes.parquet
      square_images.parquet  the square overviews taken before choosing holes (JPEG bytes per square)

See README.md ("The data") for every column. On a hidden session your policy receives the
same folder without micrographs.parquet and without the clock.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .clock import Clock

PACKAGE = Path(__file__).resolve().parent.parent
DATA_DIR = PACKAGE / "data"

# What EPU's hole finder reports for every hole when the square is set up: all known
# before a hole is shot.
HOLE_FEATURES = (
    "px_norm", "py_norm", "dist_center", "dist_edge", "log_intensity", "intensity_z_square",
    "intensity_rank_square", "log_intensity_std", "near_grid_bar", "order_norm",
    "nbr_log_intensity", "nbr_intensity_contrast", "geometry_known",
)
SQUARE_FEATURES = ("log_target_count", "all_log_intensity_mean", "near_grid_bar_fraction")
# What CryoSPARC Live reports for a micrograph about 75 s after the exposure.
LIVE_COLUMNS = ("picks", "ctf_fit_A", "defocus_um", "ice_rel", "motion_total_px", "motion_early_px")
# The expert's hand processing: known for training sessions only, never on the day.
LABEL_COLUMNS = ("curated_count", "total_count")


@dataclass
class Session:
    """One recorded session.

    On a training session every field is filled. The ``info`` a policy receives at
    ``reset`` has ``micrographs = None`` and ``clock = None``: on the day nothing has
    been shot yet and the instrument's clock is unknown."""

    name: str
    specimen: str
    squares: pd.DataFrame
    holes: pd.DataFrame
    embeddings: np.ndarray
    path: Optional[Path] = None
    micrographs: Optional[pd.DataFrame] = None
    clock: Optional[Clock] = None
    meta: dict = field(default_factory=dict)
    _hole_images: Optional[np.ndarray] = field(default=None, repr=False)
    _square_jpeg: Optional[dict] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.hole_row = {h: i for i, h in enumerate(self.holes["hole_id"])}
        self.square_row = {s: i for i, s in enumerate(self.squares["square_id"])}

    def __repr__(self) -> str:
        n = 0 if self.micrographs is None else len(self.micrographs)
        return (f"Session({self.name!r}, {self.specimen!r}: {len(self.squares)} squares, "
                f"{len(self.holes)} holes, {n} labelled micrographs)")

    @property
    def hole_images(self) -> Optional[np.ndarray]:
        """Every hole's crop at once: uint8 (n_holes, 64, 64), rows aligned with ``holes``
        (all zeros where ``has_image`` is False)."""
        if self._hole_images is None and self.path is not None and (self.path / "hole_images.npy").exists():
            self._hole_images = np.load(self.path / "hole_images.npy")
        return self._hole_images

    def hole_image(self, hole_id: str) -> Optional[np.ndarray]:
        """64x64 uint8 crop of the square overview around the hole (None if absent)."""
        imgs, row = self.hole_images, self.hole_row.get(hole_id)
        if imgs is None or row is None or not bool(self.holes["has_image"].iat[row]):
            return None
        return imgs[row]

    def square_image(self, square_id: str) -> Optional[np.ndarray]:
        """The square overview, long side 1024 px (uint8). Hole pixel coordinates x_px/y_px
        are in the full-size frame overview_width_px x overview_height_px."""
        if self._square_jpeg is None:
            f = None if self.path is None else self.path / "square_images.parquet"
            t = pd.read_parquet(f) if f is not None and f.exists() else None
            self._square_jpeg = {} if t is None else dict(zip(t["square_id"].astype(str), t["jpeg"]))
        data = self._square_jpeg.get(str(square_id))
        if data is None:
            return None
        import io

        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("L"))

    def public(self, name: Optional[str] = None) -> "Session":
        """What a policy sees at reset: no micrographs, no clock."""
        meta = {k: v for k, v in self.meta.items() if k in ("specimen", "pixel_size_A", "dose_e_per_A2", "kv")}
        return replace(self, name=name or self.name, micrographs=None, clock=None, meta=meta)

    def hole_yield(self) -> pd.Series:
        """Mean curated particles per micrograph of every hole (training sessions)."""
        if self.micrographs is None:
            raise ValueError(f"{self.name} has no labels")
        return self.micrographs.groupby("hole_id", sort=False)["curated_count"].mean().reindex(self.holes["hole_id"])


def load_session(path, *, labels: bool = True) -> Session:
    path = Path(path)
    meta = json.loads((path / "session.json").read_text())
    holes = pd.read_parquet(path / "holes.parquet")
    holes["hole_id"] = holes["hole_id"].astype(str)
    holes["square_id"] = holes["square_id"].astype(str)
    squares = pd.read_parquet(path / "squares.parquet")
    squares["square_id"] = squares["square_id"].astype(str)
    emb = np.load(path / "embeddings.npy").astype(np.float32)
    micrographs, clock = None, None
    if labels and (path / "micrographs.parquet").exists():
        micrographs = pd.read_parquet(path / "micrographs.parquet")
        for c in ("micrograph_id", "hole_id", "square_id"):
            micrographs[c] = micrographs[c].astype(str)
    if labels and meta.get("clock"):
        clock = Clock.from_dict(meta["clock"])
    return Session(name=meta["name"], specimen=meta.get("specimen", ""), squares=squares, holes=holes,
                   embeddings=emb, path=path, micrographs=micrographs, clock=clock, meta=meta)


def session_names(data_dir=DATA_DIR) -> list:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"no {data_dir}: run `python main.py download` first")
    return sorted(p.name for p in data_dir.iterdir() if (p / "session.json").exists())


def training_sessions(data_dir=DATA_DIR) -> list:
    """Every session under data/, labels included."""
    return [load_session(Path(data_dir) / n) for n in session_names(data_dir)]


# Where data/ lives: a Hugging Face dataset.
DATA_REPO = "xkronosx/cryoem-challenge-data"


def download_data(data_dir=DATA_DIR) -> Path:
    """Fetch the four training sessions into data/, unless they are already there."""
    data_dir = Path(data_dir)
    if (data_dir / "references.json").exists():
        return data_dir
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=data_dir,
                      allow_patterns=["*.parquet", "*.npy", "*.json"])
    print(f"data in {data_dir}")
    return data_dir
