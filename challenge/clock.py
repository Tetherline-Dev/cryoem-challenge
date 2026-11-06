"""The instrument clock: what every action costs, in seconds.

Each recorded session has its own clock, fitted from the time stamps EPU writes into
every micrograph's name:

* ``shot_s``           one exposure (the median gap between two exposures of one hole)
* ``near_move_s``      moving the beam to a nearby hole of the same square (image shift)
* ``far_move_s``       moving to a distant hole of the same square (a stage move), when the
                       session shows a distance effect; ``None`` means every move is near
* ``far_threshold_px`` the hole-to-hole distance, in square-overview pixels, past which a
                       move counts as far
* ``square_change_s``  moving to another grid square

The simulator charges the session's own clock. Your policy never sees it: it sees an
estimate that starts from the clock fitted on the training sessions and is refined from
the durations the day actually shows (``OnlineClock``), available as ``obs.clock``.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class Clock:
    shot_s: float = 10.0
    near_move_s: float = 15.0
    far_move_s: Optional[float] = None
    far_threshold_px: float = math.inf
    square_change_s: float = 180.0

    def move_s(self, distance_px: Optional[float]) -> float:
        """Seconds to move the beam between two holes of one square, ``distance_px`` apart
        (``None``: unknown, or the first hole after a square change, charged as near)."""
        if self.far_move_s is None or distance_px is None or distance_px < self.far_threshold_px:
            return self.near_move_s
        return self.far_move_s

    def to_dict(self) -> dict:
        d = asdict(self)
        d["far_threshold_px"] = None if not math.isfinite(self.far_threshold_px) else self.far_threshold_px
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Clock":
        thr = d.get("far_threshold_px")
        return cls(shot_s=float(d["shot_s"]), near_move_s=float(d["near_move_s"]),
                   far_move_s=None if d.get("far_move_s") is None else float(d["far_move_s"]),
                   far_threshold_px=math.inf if thr is None else float(thr),
                   square_change_s=float(d["square_change_s"]))


def fit_clock(sessions: Iterable) -> Clock:
    """Fit one clock from the time stamps of the given sessions (pooled).

    Consecutive exposures of one hole give the shot time; consecutive exposures in two
    holes of one square give the move time (less one shot), split into near and far at
    the 66th percentile of their pixel distance when far moves are slower; consecutive
    exposures in two squares give the square change. A category with too few samples
    keeps the default. Gaps over 30 minutes (breaks, refills) are ignored."""
    defaults = Clock()
    rows, positions = [], {}
    for s in sessions:
        h = s.holes
        key = dict(zip(h["hole_id"], h["sort_key"]))
        for hid, x, y in zip(h["hole_id"], h["x_px"], h["y_px"]):
            positions[(s.name, hid)] = (float(x), float(y))
        m = s.micrographs
        for ts, hid, sq in zip(m["timestamp_s"], m["hole_id"], m["square_id"]):
            if ts == ts:
                rows.append((float(ts), f"{s.name}:{key[hid]}", (s.name, hid), (s.name, sq)))
    if len(rows) < 100:
        return defaults
    rows.sort(key=lambda r: (r[0], r[1]))
    ts = np.array([r[0] for r in rows])
    holes = [r[2] for r in rows]
    codes = {h: i for i, h in enumerate(dict.fromkeys(holes))}
    hu = np.array([codes[h] for h in holes])
    sq_codes = {q: i for i, q in enumerate(dict.fromkeys(r[3] for r in rows))}
    sq = np.array([sq_codes[r[3]] for r in rows])
    dt = np.diff(ts)
    ok = (dt > 0) & (dt < 1800)
    same_hole = hu[1:] == hu[:-1]
    same_sq = (sq[1:] == sq[:-1]) & ~same_hole
    cross = ~same_hole & ~same_sq

    def med(mask, default):
        v = dt[mask & ok]
        return float(np.median(v)) if len(v) >= 10 else default

    shot = med(same_hole, defaults.shot_s)
    square_change = max(med(cross, defaults.square_change_s + shot) - shot, 0.0)
    near = max(med(same_sq, defaults.near_move_s + shot) - shot, 0.0)
    far, thresh = None, math.inf
    d_px, d_dt = [], []
    for i in np.flatnonzero(same_sq & ok):
        a, b = positions.get(holes[i]), positions.get(holes[i + 1])
        if a and b:
            d_px.append(math.hypot(a[0] - b[0], a[1] - b[1]))
            d_dt.append(float(dt[i]))
    if len(d_px) >= 50:
        d_px, d_dt = np.asarray(d_px), np.asarray(d_dt)
        thresh = float(np.quantile(d_px, 0.66))
        lo, hi = d_px < thresh, d_px >= thresh
        near = max(float(np.median(d_dt[lo])) - shot, 0.0)
        far = max(float(np.median(d_dt[hi])) - shot, 0.0)
        if far <= near:
            far, thresh = None, math.inf
    return Clock(shot_s=shot, near_move_s=near, far_move_s=far, far_threshold_px=thresh,
                 square_change_s=square_change)


@dataclass
class OnlineClock:
    """The agent's estimate of the instrument clock, ``obs.clock``.

    It starts from the clock fitted on the training sessions and learns from what the
    day shows: the duration of every exposure, square change and hole-to-hole move, with
    the move's pixel distance (hole positions are public). Each constant moves toward the
    median of the day's durations with weight n / (n + n0) and the prior is dropped once
    10 x n0 durations are in. Once 50 moves with a distance are in, moves are split into
    near and far at the distance that best separates their durations (none if far moves
    are not slower); before that the prior's split is used.

    It is a convenience, not part of the score: the simulator charges the session's own
    clock whatever your policy believes. Use it, or fit your own from ``obs.time_s``."""

    prior: Clock
    n0: float = 30.0
    refresh_every: int = 8
    _shot: list = field(default_factory=list)
    _moves: list = field(default_factory=list)       # (seconds, distance_px or None)
    _square: list = field(default_factory=list)
    _cached: Optional[Clock] = None
    _cached_n: int = -1

    def _est(self, prior: float, obs) -> float:
        if len(obs) == 0:
            return prior
        if len(obs) >= 10 * self.n0:
            return float(np.median(obs))
        w = len(obs) / (len(obs) + self.n0)
        return (1 - w) * prior + w * float(np.median(obs))

    @property
    def n_obs(self) -> int:
        return len(self._shot) + len(self._moves) + len(self._square)

    @property
    def model(self) -> Clock:
        n = self.n_obs
        if self._cached is None or n <= 50 or n - self._cached_n >= self.refresh_every:
            self._cached = self._fit()
            self._cached_n = n
        return self._cached

    def _fit(self) -> Clock:
        pm = self.prior
        secs = np.array([s for s, _ in self._moves], dtype=float)
        dist = np.array([np.inf if d is None else d for _, d in self._moves], dtype=float)
        known = np.isfinite(dist)
        thresh = pm.far_threshold_px
        far_prior = pm.far_move_s
        if known.sum() >= 50:
            d, s = dist[known], secs[known]
            best = None
            for q in np.linspace(0.1, 0.9, 17):
                t = float(np.quantile(d, q))
                lo, hi = s[d < t], s[d >= t]
                if len(lo) < 10 or len(hi) < 10:
                    continue
                sse = float(((lo - np.median(lo)) ** 2).sum() + ((hi - np.median(hi)) ** 2).sum())
                if best is None or sse < best[0]:
                    best = (sse, t, float(np.median(lo)), float(np.median(hi)))
            if best is not None and best[3] > best[2] + 1.0:
                thresh = best[1]
                far_prior = far_prior if far_prior is not None else pm.near_move_s
            else:
                thresh, far_prior = math.inf, None
        is_far = known & (dist >= thresh) if math.isfinite(thresh) else np.zeros(len(secs), dtype=bool)
        near = self._est(pm.near_move_s, secs[~is_far])
        far = self._est(far_prior, secs[is_far]) if far_prior is not None else None
        return Clock(shot_s=self._est(pm.shot_s, self._shot), near_move_s=near, far_move_s=far,
                     far_threshold_px=thresh if far is not None else math.inf,
                     square_change_s=self._est(pm.square_change_s, self._square))

    def observe_shot(self, seconds: float) -> None:
        self._shot.append(float(seconds))

    def observe_move(self, seconds: float, distance_px: Optional[float]) -> None:
        self._moves.append((float(seconds), None if distance_px is None else float(distance_px)))

    def observe_square_change(self, seconds: float) -> None:
        self._square.append(float(seconds))
