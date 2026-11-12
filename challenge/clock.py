"""The instrument clock: what every action costs, in seconds, as EPU spends it.

EPU parks the stage at a hole (the ANCHOR) and reaches every hole within its clustering radius of
the anchor by image shift, in about a second. A hole beyond the radius is a STAGE EVENT: the
stage moves, settles, re-centres on the hole and, when the session's autofocus rule makes one
due, autofocuses; that hole becomes the new anchor. So, per action:

* ``shot_s``           one exposure
* ``hop_s``            moving to a hole within ``cluster_radius_um`` of the anchor
* ``stage_s``          a stage event, at ``stage_ref_um`` from the anchor; it grows by
                       ``stage_slope_s_per_um`` for every um farther (0 where the session shows no
                       growth)
* ``autofocus_s``      added to a stage event when an autofocus is due: at every one
                       (``autofocus_every_stage``, EPU's AfterCentering) or once the hole is
                       ``autofocus_distance_um`` from the last autofocus (AfterDistance)
* ``square_change_s``  moving to another square and centring on its first hole, which becomes
                       the anchor and the autofocus position; it grows by
                       ``square_change_slope_s_per_mm`` for every mm the new square lies farther
                       than ``square_change_ref_mm`` on the grid (0 where flat)

The radius, the autofocus rule and the square image's pixel size are the session's EPU settings
(``EpuPresets``: ``session.json`` "epu", ``info.presets``), fixed before the first exposure, so
your policy knows them on every session, hidden ones included. The seconds are each category's
mean over the session's own recorded transitions, with operator breaks left out. On the recorded
sessions this clock prices the operator's own day within 0-4% of the time it really took.

The simulator charges the session's own clock. Your policy never sees it: ``obs.clock`` is an
estimate that starts from the training sessions' clock under the session's own settings
(``fit_clock``) and learns from the durations the day shows (``OnlineClock``).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

AFTER_CENTERING, AFTER_DISTANCE = "AfterCentering", "AfterDistance"
STAGE_CLUSTERING = "ClusteringWithStage"
# the operator-break rule: a duration is a pause when it is 30 min or more, or when it overruns its
# category's routine level (the median, the smallest of fewer than 3) by more than 100 s and by
# more than the level itself
MAX_GAP_S = 1800.0
PAUSE_ALLOWANCE_S = 100.0
PAUSE_RATIO = 2.0
PAUSE_MIN_N = 3
SLOPE_LEVEL = 0.95        # a distance slope is charged only when its 95% interval excludes 0
MIN_SQUARE_CHANGES = 2


@dataclass(frozen=True)
class EpuPresets:
    """A session's EPU settings that shape the clock, set before the first exposure."""

    clustering_mode: str = "ClusteringWithImageBeamShift"   # or ClusteringWithStage
    clustering_radius_um: float = 6.0
    autofocus_recurrence: str = AFTER_CENTERING             # or AfterDistance
    autofocus_distance_um: float = math.inf
    delay_after_stage_shift_s: float = math.nan
    delay_after_image_shift_s: float = math.nan
    square_pixel_nm: float = 20.0                           # x_px / y_px are pixels of this size
    epu_version: str = ""

    @property
    def um_per_px(self) -> float:
        return self.square_pixel_nm / 1000.0

    @property
    def autofocus_every_stage(self) -> bool:
        return self.autofocus_recurrence != AFTER_DISTANCE

    @classmethod
    def from_dict(cls, d: Mapping) -> "EpuPresets":
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d and d[f.name] is not None}
        return cls(**kw)


@dataclass(frozen=True)
class Clock:
    shot_s: float = 5.0
    hop_s: float = 0.3
    stage_s: float = 50.0
    autofocus_s: float = 20.0
    stage_slope_s_per_um: float = 0.0
    stage_ref_um: float = 0.0
    square_change_s: float = 110.0
    square_change_slope_s_per_mm: float = 0.0
    square_change_ref_mm: float = 0.0
    cluster_radius_um: float = 6.0
    um_per_px: float = 0.02
    autofocus_every_stage: bool = True
    autofocus_distance_um: float = math.inf
    stage_share: float = 0.13         # stage events among hole changes (the fit's)
    autofocus_share: float = 0.67     # stage events that autofocus (the fit's)

    # ----- moves -----------------------------------------------------------------------
    def in_cluster(self, distance_px: Optional[float]) -> bool:
        """Whether a hole ``distance_px`` from the anchor is reached by image shift."""
        return distance_px is not None and distance_px * self.um_per_px <= self.cluster_radius_um

    def autofocus_due(self, distance_px: Optional[float]) -> bool:
        """Whether a stage event ``distance_px`` from the last autofocus autofocuses."""
        return (self.autofocus_every_stage or distance_px is None
                or distance_px * self.um_per_px >= self.autofocus_distance_um)

    def move(self, anchor_px: Optional[float], autofocus_px: Optional[float]) -> tuple:
        """(seconds, kind) to move to a hole ``anchor_px`` pixels from the anchor and
        ``autofocus_px`` from the last autofocus: kind 'hop', 'stage' or 'stage_af'. An unknown
        distance is a stage event that autofocuses, at the reference distance."""
        if self.in_cluster(anchor_px):
            return self.hop_s, "hop"
        stage = self.stage_s
        if self.stage_slope_s_per_um and anchor_px is not None:
            stage = max(stage + self.stage_slope_s_per_um * (anchor_px * self.um_per_px - self.stage_ref_um), 0.0)
        if self.autofocus_due(autofocus_px):
            return stage + self.autofocus_s, "stage_af"
        return stage, "stage"

    def square_change_at(self, distance_mm: Optional[float] = None) -> float:
        """Seconds to change square, ``distance_mm`` on the grid from the square the stage is in
        (None: the change at its reference distance)."""
        if self.square_change_slope_s_per_mm and distance_mm is not None:
            return max(self.square_change_s + self.square_change_slope_s_per_mm
                       * (float(distance_mm) - self.square_change_ref_mm), 0.0)
        return self.square_change_s

    @property
    def stage_event_s(self) -> float:
        """A stage event at the reference distance, its autofocus at the fitted share."""
        return self.stage_s + self.autofocus_share * self.autofocus_s

    @property
    def mean_move_s(self) -> float:
        """The mean hole-to-hole move at the fitted share of stage events."""
        return (1.0 - self.stage_share) * self.hop_s + self.stage_share * self.stage_event_s

    # ----- the previous clock's names, kept so code written for it still runs ---------
    @property
    def near_move_s(self) -> float:
        return self.hop_s

    @property
    def far_move_s(self) -> float:
        return self.stage_event_s

    @property
    def far_threshold_px(self) -> float:
        return self.cluster_radius_um / self.um_per_px

    def move_s(self, distance_px: Optional[float]) -> float:
        """The previous clock's form: a move ``distance_px`` from the LAST hole, near within the
        clustering radius, else a stage event. ``obs.move_s(hole)`` prices from the anchor."""
        if distance_px is None or distance_px * self.um_per_px <= self.cluster_radius_um:
            return self.hop_s
        return self.stage_event_s

    # ----- serialisation ----------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        if not math.isfinite(d["autofocus_distance_um"]):
            d["autofocus_distance_um"] = None
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> "Clock":
        kw = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        if kw.get("autofocus_distance_um") is None:
            kw["autofocus_distance_um"] = math.inf
        return cls(**kw)

    @classmethod
    def for_presets(cls, presets: EpuPresets, **seconds) -> "Clock":
        return cls(cluster_radius_um=presets.clustering_radius_um, um_per_px=presets.um_per_px,
                   autofocus_every_stage=presets.autofocus_every_stage,
                   autofocus_distance_um=presets.autofocus_distance_um, **seconds)


# ----- fitting a clock on recorded sessions --------------------------------------------------------

def _sequence(session) -> list:
    """(time s, hole_id, square_id) of every exposure of a labelled session, in time order."""
    h, m = session.holes, session.micrographs
    key = dict(zip(h["hole_id"], h["sort_key"]))
    rows = sorted((float(t), key[hid], hid, sq) for t, hid, sq in zip(m["timestamp_s"], m["hole_id"], m["square_id"])
                  if t == t)
    return [(t, hid, sq) for t, _, hid, sq in rows]


def _positions(session) -> dict:
    h = session.holes
    return {hid: (float(x), float(y)) for hid, x, y in zip(h["hole_id"], h["x_px"], h["y_px"]) if x == x and y == y}


def _distance(positions, a, b) -> Optional[float]:
    pa, pb = positions.get(a) if a is not None else None, positions.get(b)
    if pa is None or pb is None:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def classify(seq: Sequence, positions: Mapping, presets: EpuPresets) -> list:
    """(kind, um from the anchor) of every transition of an exposure sequence: 'shot' (same hole),
    'hop' / 'stage' / 'stage_af' (another hole of the square, by the anchor rule) or 'square'."""
    rule = Clock.for_presets(presets)
    steps, anchor, autofocus = [], None, None
    for i, (_, hole, square) in enumerate(seq):
        if i == 0:
            anchor = autofocus = hole
            continue
        _, prev, prev_square = seq[i - 1]
        if square != prev_square:
            steps.append(("square", None))
            anchor = autofocus = hole
        elif hole == prev:
            steps.append(("shot", None))
        else:
            d = _distance(positions, anchor, hole)
            _, kind = rule.move(d, _distance(positions, autofocus, hole))
            steps.append((kind, None if d is None else d * presets.um_per_px))
            if kind in ("stage", "stage_af"):
                anchor = hole
            if kind == "stage_af":
                autofocus = hole
    return steps


def pause_mask(durations: Sequence[float], level: Optional[float] = None) -> np.ndarray:
    """Which durations of one category are operator breaks (left out of every mean)."""
    d = np.asarray(durations, dtype=float)
    if level is None:
        if not len(d):
            return np.zeros(0, dtype=bool)
        level = float(np.median(d)) if len(d) >= PAUSE_MIN_N else float(np.min(d))
    return (d >= MAX_GAP_S) | (d > level + max(PAUSE_ALLOWANCE_S, (PAUSE_RATIO - 1.0) * level))


def _slope(groups) -> tuple:
    """(slope, reference) of y on x over groups of (x, y), each at its own level: least squares,
    kept only when its 95% interval excludes 0 (else 0, flat)."""
    xs, ys = [], []
    for x, y in groups:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if len(x):
            xs.append(x)
            ys.append(y)
    n = int(sum(len(x) for x in xs))
    ref = float(np.mean(np.concatenate(xs))) if xs else 0.0
    dof = n - len(xs) - 1
    sxx = float(sum(((x - x.mean()) ** 2).sum() for x in xs))
    if dof < 1 or sxx <= 0.0:
        return 0.0, ref
    b = float(sum(((x - x.mean()) * (y - y.mean())).sum() for x, y in zip(xs, ys))) / sxx
    rss = float(sum((((y - y.mean()) - b * (x - x.mean())) ** 2).sum() for x, y in zip(xs, ys)))
    se = math.sqrt(rss / dof / sxx)
    from scipy.stats import t

    half = float(t.ppf(0.5 + SLOPE_LEVEL / 2.0, dof)) * se
    return (b if abs(b) > half else 0.0), ref


@dataclass
class _Fit:
    """One session's (or several sessions') transitions by category, each move beyond the shot."""

    shots: list = field(default_factory=list)
    hops: list = field(default_factory=list)
    events: list = field(default_factory=list)       # [(kind, um from the anchor or None, seconds)]
    event_sets: list = field(default_factory=list)   # the events of each session apart
    n_shot: int = 0


def _fit_session(session, presets: EpuPresets) -> _Fit:
    seq = _sequence(session)
    steps = classify(seq, _positions(session), presets)
    dt = np.diff(np.asarray([r[0] for r in seq], float))
    kinds = np.asarray([k for k, _ in steps], dtype=object)
    ok = dt > 0
    for kind in set(kinds.tolist()):
        i = np.flatnonzero((kinds == kind) & (dt > 0))
        ok[i] &= ~pause_mask(dt[i])
    shots = dt[(kinds == "shot") & ok]
    if not len(shots):
        raise ValueError(f"{session.name}: no two exposures of one hole to time a shot")
    shot = float(np.mean(shots))
    events = [(k, d, float(dt[i]) - shot) for i, (k, d) in enumerate(steps) if k in ("stage", "stage_af") and ok[i]]
    return _Fit(shots=[shot] * len(shots), hops=list(dt[(kinds == "hop") & ok] - shot), events=events,
                event_sets=[events], n_shot=len(shots))


def _stage_terms(event_sets) -> dict:
    groups, events = [], {"stage": [], "stage_af": []}
    for evs in event_sets:
        for kind in events:
            mine = [(d, y) for k, d, y in evs if k == kind]
            events[kind] += mine
            known = [(d, y) for d, y in mine if d is not None]
            if known:
                groups.append(tuple(zip(*known)))
    b, ref = _slope(groups)
    levels = {}
    for kind, evs in events.items():
        if evs:
            offset = sum(d - ref for d, _ in evs if d is not None) / len(evs)
            levels[kind] = float(np.mean([y for _, y in evs])) - b * offset
    if not levels:
        raise ValueError("no stage event to time")
    stage = levels.get("stage", levels.get("stage_af"))
    stage_af = levels.get("stage_af", stage)
    n_stage, n_af = len(events["stage"]), len(events["stage_af"])
    return {"stage_s": stage, "autofocus_s": (stage_af - stage) if n_stage and n_af else 0.0,
            "stage_slope_s_per_um": b, "stage_ref_um": ref, "n_stage": n_stage, "n_af": n_af}


def fit_clock(sessions: Iterable, presets: Optional[EpuPresets] = None) -> Clock:
    """A clock fitted on labelled sessions, applied under ``presets`` (the session it is for;
    default: the first session's). Every category is pooled over the sessions, each session's
    transitions classified under its own presets, its breaks left out; the hop comes from the
    sessions that cluster as ``presets`` does (a stage-clustering hop is a short stage move); the
    square change is the mean of their recorded changes ("square_changes" in session.json)."""
    sessions = list(sessions)
    if not sessions:
        return Clock() if presets is None else Clock.for_presets(presets)
    own = {s.name: EpuPresets.from_dict(s.meta["epu"]) for s in sessions}
    target = presets if presets is not None else own[sessions[0].name]
    fits = {s.name: _fit_session(s, own[s.name]) for s in sessions}
    shots = [v for f in fits.values() for v in f.shots]
    same = [n for n in fits if own[n].clustering_mode == target.clustering_mode]
    hop_from = same if same else list(fits)
    hops = [v for n in hop_from for v in fits[n].hops]
    stage = _stage_terms([f.events for f in fits.values()])
    n_hop = len(hops)
    if same and len(same) < len(fits):
        n_events = sum(1 for n in same for k, _, _ in fits[n].events)
    else:
        n_events = stage["n_stage"] + stage["n_af"]
    n_events_all = stage["n_stage"] + stage["n_af"]
    shot = float(np.mean(shots))
    gaps, groups = [], []
    for s in sessions:
        sc = s.meta.get("square_changes") or {}
        g, d = list(sc.get("gaps_s", [])), list(sc.get("d_mm", []))
        gaps += g
        known = [(x, y) for x, y in zip(d, g) if x is not None]
        if known:
            groups.append(tuple(zip(*known)))
    slope, ref = _slope(groups)
    square = max(float(np.mean(gaps)) - shot, 0.0) if len(gaps) >= MIN_SQUARE_CHANGES else Clock.square_change_s
    return Clock.for_presets(
        target, shot_s=shot, hop_s=float(np.mean(hops)) if hops else 0.0,
        stage_s=stage["stage_s"], autofocus_s=stage["autofocus_s"],
        stage_slope_s_per_um=stage["stage_slope_s_per_um"], stage_ref_um=stage["stage_ref_um"],
        square_change_s=square, square_change_slope_s_per_mm=slope, square_change_ref_mm=ref,
        stage_share=n_events / (n_hop + n_events) if n_hop + n_events else 0.0,
        autofocus_share=stage["n_af"] / n_events_all if n_events_all else 1.0)


# ----- the agent's estimate on the day ------------------------------------------------------------

@dataclass
class OnlineClock:
    """The agent's estimate of the instrument clock, ``obs.clock``.

    It starts from the clock fitted on the training sessions under the day's EPU settings and
    learns from what the day shows: every exposure, hop, stage event (with or without its
    autofocus, by the anchor rule) and square change. Each constant moves toward the mean of the
    day's durations with weight n / (n + n0); the distance slopes stay the prior's.

    A convenience, not part of the score: the simulator charges the session's own clock whatever
    your policy believes. Use it, or fit your own from ``obs.time_s``."""

    prior: Clock
    n0: float = 30.0
    _shot: list = field(default_factory=list)
    _hop: list = field(default_factory=list)
    _stage: list = field(default_factory=list)        # at the reference distance
    _stage_af: list = field(default_factory=list)
    _square: list = field(default_factory=list)       # at the reference distance

    def _est(self, prior: float, obs: list) -> float:
        if not obs:
            return prior
        w = len(obs) / (len(obs) + self.n0)
        return (1 - w) * prior + w * float(np.mean(obs))

    @property
    def model(self) -> Clock:
        p = self.prior
        stage = self._est(p.stage_s, self._stage)
        stage_af = self._est(p.stage_s + p.autofocus_s, self._stage_af)
        return Clock(**{**p.to_dict(), "autofocus_distance_um": p.autofocus_distance_um,
                        "shot_s": self._est(p.shot_s, self._shot), "hop_s": self._est(p.hop_s, self._hop),
                        "stage_s": stage, "autofocus_s": max(stage_af - stage, 0.0),
                        "square_change_s": self._est(p.square_change_s, self._square)})

    def observe_shot(self, seconds: float) -> None:
        self._shot.append(float(seconds))

    def observe_move(self, seconds: float, kind: str, distance_um: Optional[float] = None) -> None:
        if kind == "hop":
            self._hop.append(float(seconds))
            return
        if kind not in ("stage", "stage_af"):
            return
        p = self.prior
        level = float(seconds)
        if p.stage_slope_s_per_um and distance_um is not None:
            level -= p.stage_slope_s_per_um * (distance_um - p.stage_ref_um)
        (self._stage if kind == "stage" else self._stage_af).append(level)

    def observe_square_change(self, seconds: float, distance_mm: Optional[float] = None) -> None:
        p = self.prior
        level = float(seconds)
        if p.square_change_slope_s_per_mm and distance_mm is not None:
            level -= p.square_change_slope_s_per_mm * (distance_mm - p.square_change_ref_mm)
        self._square.append(level)
