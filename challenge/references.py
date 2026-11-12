"""The reference orders every policy is compared with.

* ``random``      a random square, then random holes in it, square by square
* ``raster``      a random first square; then always the nearest hole not yet shot, and when
                  the square is spent the nearest square: blind to quality. It ignores where
                  the stage is parked, so it leaves clusters half done and pays extra stage moves.
* ``epu_raster``  what EPU does on its own: the raster's squares, and inside a square EPU's
                  clusters - park the stage at the hole whose clustering radius covers the most
                  holes left, shoot every hole within it by image shift (nearest first), repeat.
                  Blind to quality, cheap in movement.
* ``operator``    the operator as recorded: the session's own acquisition order, replayed
* ``oracle``      knows every hole's curated yield in advance and the instrument's clock:
                  takes the best holes (by curated particles per micrograph) until they hold
                  the target, and visits them cluster by cluster, the squares by particles per
                  second. A strong reference, not a proven optimum.

These read the simulator's state directly (the oracle reads the labels), so they run
only where the labels are: on your machine for the training sessions, on ours for the
hidden ones. ``random``, ``raster`` and ``epu_raster`` are averaged over ten seeds (their
first square).
"""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np

from .clock import Clock
from .data import Session
from .policy import GoToSquare, ShootHole
from .simulator import EpisodeResult, Simulator

REFERENCES = ("random", "raster", "epu_raster", "operator", "oracle")
SEEDS = {"random": tuple(range(10)), "raster": tuple(range(10)), "epu_raster": tuple(range(10)), "operator": (0,),
         "oracle": (0,)}


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

    def _square(self, sim: Simulator):
        if self.centroids is None:
            sq = sim.session.squares
            self.centroids = {s: (float(x), float(y)) for s, x, y in
                              zip(sq["square_id"], sq["centroid_x_um"], sq["centroid_y_um"])
                              if math.isfinite(x) and math.isfinite(y)}
        cands = [s for s in sim.square_ids if sim.left[s] > 0]
        previous = sim.square_of[sim.last_hole] if sim.last_hole else None
        here = self.centroids.get(previous) if previous else None
        if here is not None:
            cands.sort(key=lambda s: (math.hypot(self.centroids[s][0] - here[0], self.centroids[s][1] - here[1])
                                      if s in self.centroids else math.inf))
            return GoToSquare(cands[0])
        return GoToSquare(self.rng.choice(cands))

    def _next_hole(self, sim: Simulator, ordered: list) -> str:
        here = sim.xy.get(self.hole) if self.hole else None
        if here is not None:
            def gap(h):
                p = sim.xy.get(h)
                return math.hypot(p[0] - here[0], p[1] - here[1]) if p is not None else math.inf
            return min(ordered, key=gap)
        return self.rng.choice(ordered)

    def act(self, sim: Simulator):
        if sim.current_square is None:
            return self._square(sim)
        available = set(sim.holes_left(sim.current_square))
        if self.hole in available:
            return ShootHole(self.hole)
        ordered = sorted(available, key=sim.sort_key.__getitem__)
        self.hole = self._next_hole(sim, ordered)
        return ShootHole(self.hole)


class _EpuRaster(_Raster):
    """The raster's squares; inside a square, EPU's clusters (anchor, its radius, next anchor)."""

    def __init__(self, seed: int):
        super().__init__(seed)
        self.anchor: Optional[str] = None
        self.anchor_square: Optional[str] = None

    def _next_hole(self, sim: Simulator, ordered: list) -> str:
        placed = [h for h in ordered if h in sim.xy]
        if not placed:
            return super()._next_hole(sim, ordered)
        radius_px = sim.clock.cluster_radius_um / sim.clock.um_per_px
        here = sim.xy.get(self.hole) if self.hole else None

        def gap(h):
            return math.hypot(sim.xy[h][0] - here[0], sim.xy[h][1] - here[1]) if here is not None else 0.0

        def within(centre):
            return [h for h in placed if math.hypot(sim.xy[h][0] - centre[0], sim.xy[h][1] - centre[1]) <= radius_px]

        anchor = sim.xy.get(self.anchor) if self.anchor is not None and self.anchor_square == sim.current_square else None
        inside = within(anchor) if anchor is not None else []
        if inside:
            return min(inside, key=gap)
        covers = {h: len(within(sim.xy[h])) for h in placed}
        most = max(covers.values())
        hole = min((h for h in placed if covers[h] == most), key=gap)
        self.anchor, self.anchor_square = hole, sim.current_square
        return hole


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


def _cover(sim: Simulator, holes: list, radius_px: float) -> list:
    """Greedy clusters over ``holes`` (one square's): repeatedly the hole whose clustering radius
    covers the most holes left (ties: the smallest sort key), [(anchor, [holes]), ...]."""
    placed = sorted((h for h in holes if h in sim.xy), key=sim.sort_key.__getitem__)
    groups = []
    if placed:
        xy = np.array([sim.xy[h] for h in placed])
        near = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1]) <= radius_px
        left = np.ones(len(placed), dtype=bool)
        while left.any():
            counts = np.where(left, near[:, left].sum(axis=1), -1)
            best = int(np.argmax(counts))
            group = np.flatnonzero(near[best] & left)
            groups.append((placed[best], [placed[i] for i in group]))
            left[group] = False
    groups += [(h, [h]) for h in holes if h not in sim.xy]
    return groups


class _Oracle:
    """The true best holes until the target, cluster by cluster, the squares by gain per second."""

    def __init__(self, seed: int = 0):
        pass

    def act(self, sim: Simulator):
        clock: Clock = sim.clock
        radius_px = clock.cluster_radius_um / clock.um_per_px
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

        here = [c[1] for c in worth if c[0] == sim.current_square]
        if here:
            if sim.last_hole is not None and sim.anchor is not None:
                inside = [h for h in here if clock.in_cluster(sim.distance_px(sim.anchor, h))]
                if inside:
                    def gap(h):
                        d = sim.distance_px(sim.last_hole, h)
                        return (math.inf if d is None else d, sim.sort_key[h])
                    return ShootHole(min(inside, key=gap))
            # the next anchor: the hole covering the most of the square's best holes, the cheapest move
            groups = _cover(sim, here, radius_px)
            most = max(len(g) for _, g in groups)
            heads = [a for a, g in groups if len(g) == most]
            return ShootHole(min(heads, key=lambda h: (sim.move_cost(h)[0], sim.sort_key[h])))

        by_square: dict = {}
        for c in worth:
            by_square.setdefault(c[0], []).append(c)

        def rate(square, items):
            gain = sum(max(v, 0.0) * n for _, _, v, n in items)
            groups = _cover(sim, [h for _, h, _, _ in items], radius_px)
            secs = (clock.square_change_at(sim.square_distance_mm(sim.stage_square, square))
                    + sum(n for _, _, _, n in items) * clock.shot_s
                    + (len(groups) - 1) * clock.stage_event_s + (len(items) - len(groups)) * clock.hop_s)
            return gain / max(secs, 1e-9)

        return GoToSquare(max(by_square, key=lambda s: (rate(s, by_square[s]), s)))


_CLASSES = {"random": _Random, "raster": _Raster, "epu_raster": _EpuRaster, "operator": _Operator, "oracle": _Oracle}


def run_reference(name: str, session: Session, clock_prior: Clock, seed: int = 0) -> EpisodeResult:
    sim = Simulator(session, clock_prior=clock_prior)
    ref = _CLASSES[name](seed)
    while not sim.done:
        sim.step(ref.act(sim))
    return sim.result()


def reference_hours(session: Session, clock_prior: Optional[Clock] = None) -> dict:
    """Hours to target of every reference on a labelled session (seed means)."""
    prior = clock_prior or Clock.for_presets(session.presets)
    out = {}
    for name in REFERENCES:
        runs = [run_reference(name, session, prior, seed) for seed in SEEDS[name]]
        out[name] = sum(r.hours for r in runs) / len(runs)
    return out
