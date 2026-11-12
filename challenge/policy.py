"""What a policy sees and what it may do.

A policy is asked for one decision at a time:

* ``GoToSquare(square_id)``  move the stage to a grid square. Required whenever
  ``obs.current_square`` is None: at the start, and after the current square runs out
  of selected holes.
* ``ShootHole(hole_id)``     shoot a hole of the current square: every exposure of the
  operator's template there (``holes.n_micrographs``, 3 to 7), one after another. The
  hole is then spent.

Its results come back late, as they do from CryoSPARC Live: each micrograph's outputs
arrive 75 instrument-seconds after its exposure, in ``obs.new_results``. The curated
count, the score, never arrives.

Moves cost what EPU spends (challenge/clock.py): ``obs.anchor_hole`` is where the stage is
parked; a hole within the clustering radius of it is a quick image-shift hop, any other a stage
event that parks the stage there. ``obs.move_s(hole_id)`` and ``obs.move_kind(hole_id)`` price a
move by your clock estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Mapping, Optional, Union

import numpy as np

from .clock import Clock
from .data import LIVE_COLUMNS, Session


@dataclass(frozen=True)
class GoToSquare:
    square_id: str


@dataclass(frozen=True)
class ShootHole:
    hole_id: str


Action = Union[GoToSquare, ShootHole]


@dataclass(frozen=True)
class Result:
    """CryoSPARC Live's report on one micrograph, as it arrives on the day."""

    micrograph_id: str
    hole_id: str
    square_id: str
    taken_at_s: float        # instrument time of the exposure
    arrived_at_s: float      # instrument time the report reached you (taken_at_s + 75)
    picks: float             # particles the blob picker found
    ctf_fit_A: float         # resolution (Å) to which the CTF fits the Thon rings: lower is better
    defocus_um: float        # measured defocus (µm)
    ice_rel: float           # relative ice thickness
    motion_total_px: float   # total beam-induced motion (px)
    motion_early_px: float   # motion in the first frames (px)


RESULT_FIELDS = tuple(f.name for f in fields(Result))
assert RESULT_FIELDS[5:] == LIVE_COLUMNS


@dataclass
class Update:
    """What changed after one action: the message the simulator sends to the policy's
    side (in-process locally, over a pipe when graded). Plain data only."""

    time_s: float
    current_square: Optional[str]
    beam_hole: Optional[str]
    spent: list            # holes shot by the last action
    new_results: list      # Result fields as tuples, in arrival order
    pending: dict          # hole_id -> exposures taken whose results have not arrived
    clock: dict            # the agent's clock estimate (Clock fields)
    done: bool
    anchor_hole: Optional[str] = None      # where the stage is parked (None right after a square change)
    autofocus_hole: Optional[str] = None   # where the last autofocus was
    stage_square: Optional[str] = None     # the square the stage is in (also once it is spent)

    def to_json(self) -> dict:
        return {"time_s": self.time_s, "current_square": self.current_square, "beam_hole": self.beam_hole,
                "spent": list(self.spent), "new_results": [list(r) for r in self.new_results],
                "pending": dict(self.pending), "clock": dict(self.clock), "done": self.done,
                "anchor_hole": self.anchor_hole, "autofocus_hole": self.autofocus_hole,
                "stage_square": self.stage_square}

    @classmethod
    def from_json(cls, d: dict) -> "Update":
        return cls(time_s=d["time_s"], current_square=d["current_square"], beam_hole=d["beam_hole"],
                   spent=d["spent"], new_results=[tuple(r) for r in d["new_results"]],
                   pending=d["pending"], clock=d["clock"], done=d["done"], anchor_hole=d.get("anchor_hole"),
                   autofocus_hole=d.get("autofocus_hole"), stage_square=d.get("stage_square"))


class Observation:
    """The state of the session at a decision, as the policy may know it.

    Attributes
    ----------
    time_s          instrument seconds since the session started
    current_square  the square the stage is at (None: you must GoToSquare)
    beam_hole       the last hole shot in this square (None right after a square change)
    anchor_hole     where the stage is parked: holes within the clustering radius of it are hops
    autofocus_hole  where the last autofocus was (AfterDistance sessions autofocus by distance from it)
    stage_square    the square the stage is in, also after its holes run out (square changes
                    are priced from it)
    new_results     Results that arrived since your previous decision
    pending         hole_id -> exposures taken whose results are still on their way
    remaining       bool array over info.holes rows: True for holes not shot yet (read-only)
    clock           Clock: your estimate of what actions cost (prior + the day's durations)
    info            the Session (no labels) you received at reset
    """

    __slots__ = ("time_s", "current_square", "beam_hole", "anchor_hole", "autofocus_hole", "stage_square",
                 "new_results", "pending", "remaining", "clock", "info", "_state")

    def squares_left(self) -> list:
        """Squares that still have holes to shoot, in info.squares order."""
        left = self._state.left
        return [s for s in self.info.squares["square_id"] if left[s] > 0]

    def holes_left(self, square_id: Optional[str] = None) -> list:
        """Holes not shot yet in a square (default: the current one), in info.holes order."""
        sq = self.current_square if square_id is None else square_id
        if sq is None:
            return []
        rows = self._state.rows_of_square[sq]
        rem = self.remaining
        ids = self._state.hole_ids
        return [ids[r] for r in rows if rem[r]]

    def distance_px(self, a: Optional[str], b: str) -> Optional[float]:
        """Pixel distance between two holes of one square (None across squares or unknown)."""
        return self._state.distance_px(a, b)

    def distance_um(self, a: Optional[str], b: str) -> Optional[float]:
        """Distance between two holes of one square in um (the square image's pixel size)."""
        d = self._state.distance_px(a, b)
        return None if d is None else d * self.clock.um_per_px

    def move(self, hole_id: str) -> tuple:
        """(seconds, kind), by your clock estimate, to bring the beam to a hole of the current
        square from where the stage is: kind 'entry' (the first hole after a square change: its
        move is part of the change), 'none' (the beam is there), 'hop', 'stage' or 'stage_af'."""
        if self.beam_hole is None:
            return 0.0, "entry"
        if hole_id == self.beam_hole:
            return 0.0, "none"
        st = self._state
        return self.clock.move(st.distance_px(self.anchor_hole, hole_id), st.distance_px(self.autofocus_hole, hole_id))

    def move_s(self, hole_id: str) -> float:
        """Seconds (by your clock estimate) to bring the beam to a hole of the current square."""
        return self.move(hole_id)[0]

    def move_kind(self, hole_id: str) -> str:
        """'entry', 'none', 'hop', 'stage' or 'stage_af': see ``move``."""
        return self.move(hole_id)[1]

    def square_change_s(self, square_id: str) -> float:
        """Seconds (by your clock estimate) to change to a square from the one the stage is in."""
        return self.clock.square_change_at(self._state.square_distance_mm(self.stage_square, square_id))

    def cost_s(self, hole_id: str) -> float:
        """Seconds (by your clock estimate) to shoot a hole of the current square from where
        the beam is: the move plus every exposure of its template."""
        n = self._state.n_shots[self.info.hole_row[hole_id]]
        return self.move_s(hole_id) + n * self.clock.shot_s


class ObservationState:
    """Builds Observations from Updates. Identical on your machine and in grading."""

    def __init__(self, info: Session):
        self.info = info
        h = info.holes
        self.hole_ids = list(h["hole_id"])
        self.square_of = dict(zip(h["hole_id"], h["square_id"]))
        self.n_shots = h["n_micrographs"].to_numpy()
        self.xy = {hid: (float(x), float(y)) for hid, x, y in zip(h["hole_id"], h["x_px"], h["y_px"])
                   if x == x and y == y}
        self.rows_of_square = {s: [] for s in info.squares["square_id"]}
        for i, s in enumerate(h["square_id"]):
            self.rows_of_square[s].append(i)
        self.left = {s: len(r) for s, r in self.rows_of_square.items()}
        q = info.squares
        self.atlas = {sid: (float(x), float(y)) for sid, x, y in zip(q["square_id"], q["stage_x_um"], q["stage_y_um"])
                      if x == x and y == y}
        self._remaining = np.ones(len(h), dtype=bool)
        self._view = self._remaining.view()
        self._view.flags.writeable = False

    def distance_px(self, a: Optional[str], b: str) -> Optional[float]:
        if a is None or self.square_of.get(a) != self.square_of.get(b):
            return None
        pa, pb = self.xy.get(a), self.xy.get(b)
        if pa is None or pb is None:
            return None
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def square_distance_mm(self, a: Optional[str], b: str) -> Optional[float]:
        pa, pb = self.atlas.get(a) if a is not None else None, self.atlas.get(b)
        if pa is None or pb is None:
            return None
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / 1000.0

    def apply(self, u: Update) -> Observation:
        for hid in u.spent:
            r = self.info.hole_row[hid]
            if self._remaining[r]:
                self._remaining[r] = False
                self.left[self.square_of[hid]] -= 1
        obs = Observation()
        obs.time_s = float(u.time_s)
        obs.current_square = u.current_square
        obs.beam_hole = u.beam_hole
        obs.anchor_hole = u.anchor_hole
        obs.autofocus_hole = u.autofocus_hole
        obs.stage_square = u.stage_square
        obs.new_results = tuple(Result(*r) for r in u.new_results)
        obs.pending = MappingProxyType(dict(u.pending))
        obs.remaining = self._view
        obs.clock = Clock.from_dict(u.clock)
        obs.info = self.info
        obs._state = self
        return obs


class Policy:
    """Subclass this. ``reset`` is called once per session, ``act`` once per decision."""

    def reset(self, info: Session) -> None:
        """A new session starts. ``info`` has the squares, holes, embeddings and images,
        and no labels."""

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError
