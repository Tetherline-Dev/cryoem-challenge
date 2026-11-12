"""The replay of one recorded session.

Every hole of the session was really shot, and every exposure's outcome is on record. The
simulator replays them in whatever order your policy chooses, charges the instrument time
the session's own clock says each action costs, and stops as soon as the micrographs
collected hold 75% of the curated particles the whole session yielded. Your score is the
instrument time it took.

Moves are charged as EPU spends them (challenge/clock.py): from where the stage is parked (the
anchor), a hole within the clustering radius is an image-shift hop, a hole beyond it a stage
event that becomes the new anchor; a square change includes centring on the new square's first
hole, which becomes the anchor.

The simulator holds the labels; your policy never sees them. It talks to the policy only
through ``Update`` messages (challenge/policy.py), the same way on your machine and in
grading.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

from .clock import Clock, OnlineClock
from .data import LIVE_COLUMNS, Session
from .policy import GoToSquare, ShootHole, Update

TARGET_FRACTION = 0.75     # of the session's curated particles
LATENCY_S = 75.0           # CryoSPARC Live's turnaround, instrument seconds


class PolicyError(RuntimeError):
    """The policy asked for something the microscope cannot do."""


@dataclass
class EpisodeResult:
    session: str
    hours: float
    completed: bool
    micrographs: int
    holes: int
    square_changes: int
    curve: list = field(default_factory=list)    # (hours, fraction of the target) after every hole
    compute_s: float = 0.0                       # the policy's own thinking time (wall clock)
    stage_events: int = 0                        # moves beyond the clustering radius
    seconds: dict = field(default_factory=dict)  # instrument seconds by kind: shot, hop, stage, stage_af, square

    def to_dict(self) -> dict:
        return asdict(self)


class Simulator:
    def __init__(self, session: Session, *, clock_prior: Clock, target_fraction: float = TARGET_FRACTION,
                 latency_s: float = LATENCY_S):
        if session.micrographs is None or session.clock is None:
            raise ValueError(f"{session.name}: the simulator needs the labelled session")
        self.session = session
        self.clock = session.clock
        self.clock_prior = clock_prior
        self.latency_s = float(latency_s)
        h, m = session.holes, session.micrographs
        self.hole_ids = list(h["hole_id"])
        self.square_ids = list(session.squares["square_id"])
        self.square_of = dict(zip(h["hole_id"], h["square_id"]))
        self.sort_key = dict(zip(h["hole_id"], h["sort_key"]))
        self.xy = {hid: (float(x), float(y)) for hid, x, y in zip(h["hole_id"], h["x_px"], h["y_px"])}
        q = session.squares
        self.atlas = {sid: (float(x), float(y)) for sid, x, y in zip(q["square_id"], q["stage_x_um"], q["stage_y_um"])
                      if x == x and y == y}
        self.events_log = None          # set to a list to record every action (grading's trace)
        self.holes_of = {s: [] for s in self.square_ids}
        for hid, s in zip(h["hole_id"], h["square_id"]):
            self.holes_of[s].append(hid)
        m = m.sort_values(["hole_id", "shot_index"], kind="stable")
        cols = ("micrograph_id", "hole_id", "square_id") + LIVE_COLUMNS
        self.shots = {hid: [] for hid in self.hole_ids}
        for row in zip(*(m[c] for c in cols), m["curated_count"], m["timestamp_s"]):
            self.shots[row[1]].append(row)
        total = 0
        for hid in self.hole_ids:
            for r in self.shots[hid]:
                total += float(r[-2])
        self.total = float(total)
        self.target = float(target_fraction) * self.total
        self.reset()

    # ----- state -----------------------------------------------------------------
    def reset(self) -> Update:
        self.queue = {hid: list(self.shots[hid]) for hid in self.hole_ids}
        self.left = {s: len(v) for s, v in self.holes_of.items()}
        self.current_square: Optional[str] = None
        self.last_hole: Optional[str] = None
        self.anchor: Optional[str] = None          # where the last stage move landed
        self.autofocus_at: Optional[str] = None    # where the last autofocus was
        self.stage_square: Optional[str] = None    # the square the stage is in
        self.stage_events = 0
        self.seconds = {k: 0.0 for k in ("shot", "hop", "stage", "stage_af", "square")}
        self.time_s = 0.0
        self.square_changes = 0
        self.visits = 0
        self.n_micrographs = 0
        self.collected = 0.0
        self.ledger: list = []          # (reveal time, result tuple)
        self._shown = 0                 # ledger entries already delivered
        self._square_streak = 0
        self._spent: list = []
        self.agent_clock = OnlineClock(self.clock_prior)
        self.curve: list = []
        return self._update()

    @property
    def done(self) -> bool:
        return self.collected >= self.target or all(v == 0 for v in self.left.values())

    def holes_left(self, square_id: str) -> list:
        return [h for h in self.holes_of[square_id] if self.queue[h]]

    def distance_px(self, a: Optional[str], b: str) -> Optional[float]:
        if a is None or self.square_of[a] != self.square_of[b]:
            return None
        pa, pb = self.xy.get(a), self.xy.get(b)
        if pa is None or pb is None:
            return None
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

    def square_distance_mm(self, a: Optional[str], b: str) -> Optional[float]:
        """Grid distance between two squares in mm (None before the first square or unknown)."""
        pa, pb = self.atlas.get(a) if a is not None else None, self.atlas.get(b)
        if pa is None or pb is None:
            return None
        return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) / 1000.0

    def move_cost(self, hid: str, clock=None) -> tuple:
        """(seconds, kind) of moving to a hole of the current square from where the stage is:
        'entry' (the square's first hole: part of the square change), 'hop', 'stage', 'stage_af',
        or 'none' (the beam is there)."""
        clock = clock or self.clock
        if self.last_hole is None:
            return 0.0, "entry"
        if hid == self.last_hole:
            return 0.0, "none"
        return clock.move(self.distance_px(self.anchor, hid), self.distance_px(self.autofocus_at, hid))

    # ----- transitions -------------------------------------------------------------
    def step(self, action) -> Update:
        if self.done:
            raise PolicyError("the session is over")
        if isinstance(action, GoToSquare):
            sq = str(action.square_id)
            if sq not in self.left:
                raise PolicyError(f"unknown square {sq!r}")
            if self.left[sq] == 0:
                raise PolicyError(f"square {sq} has no holes left")
            if sq != self.current_square:
                if self.current_square is not None:
                    self.square_changes += 1
                d_mm = self.square_distance_mm(self.stage_square, sq)
                change = self.clock.square_change_at(d_mm)
                self.time_s += change
                self.seconds["square"] += change
                self.agent_clock.observe_square_change(change, d_mm)
                self._log("square", sq, None, change, d_mm)
                self.current_square = sq
                self.stage_square = sq
                self.last_hole = self.anchor = self.autofocus_at = None
            self._square_streak += 1
            if self._square_streak > 4:
                raise PolicyError("the policy keeps choosing squares without shooting a hole")
            return self._update()
        if isinstance(action, ShootHole):
            hid = str(action.hole_id)
            if self.current_square is None:
                raise PolicyError("choose a square (GoToSquare) before shooting a hole")
            if hid not in self.queue:
                raise PolicyError(f"unknown hole {hid!r}")
            if self.square_of[hid] != self.current_square:
                raise PolicyError(f"hole {hid} is not in the current square {self.current_square}")
            if not self.queue[hid]:
                raise PolicyError(f"hole {hid} was already shot")
            self._square_streak = 0
            self._shoot(hid)
            if self.done or self.left[self.current_square] == 0:
                self.current_square = None
            return self._update()
        raise PolicyError(f"not an action: {action!r} (return GoToSquare or ShootHole)")

    def _shoot(self, hid: str) -> None:
        move, kind = self.move_cost(hid)
        if kind == "entry":
            self.anchor = self.autofocus_at = hid
        elif kind != "none":
            d = self.distance_px(self.anchor, hid)
            self.time_s += move
            self.seconds[kind] += move
            self.agent_clock.observe_move(move, kind, None if d is None else d * self.clock.um_per_px)
            if kind in ("stage", "stage_af"):
                self.anchor = hid
                self.stage_events += 1
            if kind == "stage_af":
                self.autofocus_at = hid
        self.last_hole = hid
        self.visits += 1
        before = self.collected
        self._log("hole", self.square_of[hid], hid, move, kind)
        for row in self.queue[hid]:
            self.time_s += self.clock.shot_s
            self.seconds["shot"] += self.clock.shot_s
            self.agent_clock.observe_shot(self.clock.shot_s)
            self.n_micrographs += 1
            self.collected += float(row[-2])
            result = (row[0], row[1], row[2], self.time_s, self.time_s + self.latency_s) + tuple(
                float(v) for v in row[3:3 + len(LIVE_COLUMNS)])
            self.ledger.append((self.time_s + self.latency_s, result))
        self.queue[hid] = []
        self.left[self.square_of[hid]] -= 1
        if self.events_log is not None:
            self.events_log[-1].update(particles=self.collected - before, t_end_h=self.time_s / 3600.0)
        self._spent.append(hid)
        self.curve.append((self.time_s / 3600.0, min(self.collected / self.target, 1.0)))

    def _update(self) -> Update:
        new = []
        while self._shown < len(self.ledger) and self.ledger[self._shown][0] <= self.time_s:
            new.append(self.ledger[self._shown][1])
            self._shown += 1
        pending: dict = {}
        for _, r in self.ledger[self._shown:]:
            pending[r[1]] = pending.get(r[1], 0) + 1
        c = self.agent_clock.model
        spent, self._spent = self._spent, []
        here = self.current_square is not None
        return Update(time_s=self.time_s, current_square=self.current_square,
                      beam_hole=self.last_hole if here else None,
                      spent=spent, new_results=new, pending=pending, clock=c.to_dict(), done=self.done,
                      anchor_hole=self.anchor if here else None, autofocus_hole=self.autofocus_at if here else None,
                      stage_square=self.stage_square)

    def _log(self, what: str, square: str, hole: Optional[str], seconds: float, detail) -> None:
        if self.events_log is not None:
            self.events_log.append({"t_h": self.time_s / 3600.0, "action": what, "square": square, "hole": hole,
                                    "move_s": seconds, "kind": detail if what == "hole" else "square",
                                    "d_mm": detail if what == "square" else None, "particles": 0.0})

    def result(self) -> EpisodeResult:
        return EpisodeResult(session=self.session.name, hours=self.time_s / 3600.0,
                             completed=self.collected >= self.target, micrographs=self.n_micrographs,
                             holes=self.visits, square_changes=self.square_changes, curve=list(self.curve),
                             stage_events=self.stage_events, seconds=dict(self.seconds))
