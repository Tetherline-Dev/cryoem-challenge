"""The reference orders every policy is compared with.

* ``random``    a random square, then random holes in it, square by square
* ``raster``    a random first square; then always the nearest hole not yet shot, and when
                the square is spent the nearest square: blind to quality, cheap in movement.
                An expert cryo-EM practitioner called it a good stand-in for an operator.
* ``operator``  the operator as recorded: the session's own acquisition order, replayed
* ``oracle``    knows every hole's curated yield in advance and the instrument's clock:
                takes the best holes (by curated particles per micrograph) until they hold
                the target, along a least-time path. A strong reference, not a proven optimum.

These read the simulator's state directly (the oracle reads the labels), so they run
only where the labels are: on your machine for the training sessions, on ours for the
hidden ones. ``random`` and ``raster`` are averaged over two seeds.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from .clock import Clock
from .data import Session
from .policy import GoToSquare, ShootHole
from .simulator import EpisodeResult, Simulator

REFERENCES = ("random", "raster", "operator", "oracle")
SEEDS = {"random": (0, 1), "raster": (0, 1), "operator": (0,), "oracle": (0,)}


class _Random:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def act(self, sim: Simulator):
        if sim.current_square is None:
            return GoToSquare(self.rng.choice([s for s in sim.square_ids if sim.left[s] > 0]))
        return ShootHole(self.rng.choice(sim.holes_left(sim.current_square)))


class _Raster:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.hole: Optional[str] = None
        self.centroids: Optional[dict] = None

    def act(self, sim: Simulator):
        if self.centroids is None:
            sq = sim.session.squares
            self.centroids = {s: (float(x), float(y)) for s, x, y in
                              zip(sq["square_id"], sq["centroid_x_um"], sq["centroid_y_um"])
                              if math.isfinite(x) and math.isfinite(y)}
        if sim.current_square is None:
            cands = [s for s in sim.square_ids if sim.left[s] > 0]
            previous = sim.square_of[sim.last_hole] if sim.last_hole else None
            here = self.centroids.get(previous) if previous else None
            if here is not None:
                cands.sort(key=lambda s: (math.hypot(self.centroids[s][0] - here[0], self.centroids[s][1] - here[1])
                                          if s in self.centroids else math.inf))
                return GoToSquare(cands[0])
            return GoToSquare(self.rng.choice(cands))
        available = set(sim.holes_left(sim.current_square))
        if self.hole in available:
            return ShootHole(self.hole)
        ordered = sorted(available, key=sim.sort_key.__getitem__)
        here = sim.xy.get(self.hole) if self.hole else None
        if here is not None:
            def gap(h):
                p = sim.xy.get(h)
                return math.hypot(p[0] - here[0], p[1] - here[1]) if p is not None else math.inf
            self.hole = min(ordered, key=gap)
        else:
            self.hole = self.rng.choice(ordered)
        return ShootHole(self.hole)


class _Operator:
    def __init__(self, seed: int = 0):
        self.order: Optional[list] = None
        self.next = 0

    def act(self, sim: Simulator):
        if self.order is None:
            first = {h: min((r[-1] for r in sim.shots[h] if r[-1] == r[-1]), default=math.inf)
                     for h in sim.hole_ids}
            self.order = sorted(first, key=lambda h: (first[h], sim.sort_key[h]))
        while self.next < len(self.order) and not sim.queue[self.order[self.next]]:
            self.next += 1
        hid = self.order[min(self.next, len(self.order) - 1)]
        if sim.current_square != sim.square_of[hid]:
            return GoToSquare(sim.square_of[hid])
        return ShootHole(hid)


class _Oracle:
    """Ideal path: the true best holes until the target, along a nearest-neighbour path."""

    def __init__(self, seed: int = 0):
        pass

    def act(self, sim: Simulator):
        clock = sim.clock
        needed = max(sim.target - sim.collected, 1e-9)
        cands = []
        for s in sim.square_ids:
            for h in sim.holes_left(s):
                q = sim.queue[h]
                cands.append((s, h, float(sum(r[-2] for r in q) / len(q)), max(len(q), 1)))
        ranked = sorted(cands, key=lambda c: (-c[2], sim.sort_key[c[1]]))
        worth, got = [], 0.0
        for c in ranked:
            worth.append(c)
            got += max(c[2], 0.0) * c[3]
            if got >= needed:
                break

        def seconds(h, after_switch=False):
            n = max(len(sim.queue[h]), 1)
            if after_switch:
                move = clock.near_move_s
            elif h != sim.last_hole:
                move = clock.move_s(sim.distance_px(sim.last_hole, h))
            else:
                move = 0.0
            return (move + n * clock.shot_s) / n

        here = [c for c in worth if c[0] == sim.current_square]
        if here:
            def dist(h):
                d = sim.distance_px(sim.last_hole, h)
                return math.inf if d is None else d
            best = min(here, key=lambda c: (seconds(c[1]), dist(c[1]), -c[2], sim.sort_key[c[1]]))
            return ShootHole(best[1])
        by_square: dict = {}
        for c in worth:
            by_square.setdefault(c[0], []).append(c)

        def rate(items):
            gain = sum(max(v, 0.0) * n for _, _, v, n in items)
            secs = clock.square_change_s + sum(seconds(h, after_switch=True) * n for _, h, _, n in items)
            return gain / max(secs, 1e-9)

        return GoToSquare(max(by_square, key=lambda s: (rate(by_square[s]), s)))


_CLASSES = {"random": _Random, "raster": _Raster, "operator": _Operator, "oracle": _Oracle}


def run_reference(name: str, session: Session, clock_prior: Clock, seed: int = 0) -> EpisodeResult:
    sim = Simulator(session, clock_prior=clock_prior)
    ref = _CLASSES[name](seed)
    while not sim.done:
        sim.step(ref.act(sim))
    return sim.result()


def reference_hours(session: Session, clock_prior: Optional[Clock] = None) -> dict:
    """Hours to target of every reference on a labelled session (seed means)."""
    prior = clock_prior or Clock()
    out = {}
    for name in REFERENCES:
        runs = [run_reference(name, session, prior, seed) for seed in SEEDS[name]]
        out[name] = sum(r.hours for r in runs) / len(runs)
    return out
