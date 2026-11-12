"""Running a policy: one replay, leave-one-session-out validation, training.

A policy is a Python file (my_policy.py) that defines two functions:

    def train(sessions: list[Session], out_dir: Path) -> None
        # fit on these labelled sessions; write whatever you need into out_dir
    def load(out_dir: Path) -> Policy
        # rebuild the policy from out_dir

Grading calls only ``load``, on the artefacts you submit, then plays each hidden session.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from .clock import Clock, fit_clock
from .data import Session, training_sessions
from .policy import ObservationState
from .simulator import EpisodeResult, Simulator


def artifacts_dir(policy_file) -> Path:
    """Where `python main.py train` puts a policy's artefacts: artifacts/ for my_policy.py,
    artifacts_<name>/ for any other policy file."""
    f = Path(policy_file)
    return f.parent / ("artifacts" if f.stem == "my_policy" else f"artifacts_{f.stem}")


def load_policy_module(path):
    """Import a policy file (my_policy.py) as a fresh module, under its own name, so that
    objects of the classes it defines can be pickled into the artefacts and loaded back."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"no policy file at {file}")
    folder = str(file.parent.resolve())
    if folder not in sys.path:
        sys.path.insert(0, folder)
    name = file.stem
    taken = sys.modules.get(name)
    if taken is not None and not getattr(taken, "_challenge_policy", False) and \
            Path(getattr(taken, "__file__", "") or "").resolve() != file.resolve():
        name = f"_policy_{abs(hash(str(file.resolve())))}"  # the name belongs to another module
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    module._challenge_policy = True
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if taken is not None and name == file.stem:
            sys.modules[name] = taken
        else:
            sys.modules.pop(name, None)
        raise
    for fn in ("train", "load"):
        if not callable(getattr(module, fn, None)):
            raise AttributeError(f"{file} must define {fn}(); see my_policy.py")
    return module


def run_policy(policy, session: Session, clock_prior: Clock, *, name: Optional[str] = None,
               time_budget_s: Optional[float] = None) -> EpisodeResult:
    """Replay one labelled session with a policy object. ``time_budget_s`` caps the
    policy's own thinking time (reset + every act), not the simulated hours."""
    sim = Simulator(session, clock_prior=clock_prior)
    info = session.public(name)
    state = ObservationState(info)
    spent = 0.0
    t0 = time.perf_counter()
    policy.reset(info)
    spent += time.perf_counter() - t0
    u = sim.reset()
    obs = state.apply(u)
    while not u.done:
        t0 = time.perf_counter()
        action = policy.act(obs)
        spent += time.perf_counter() - t0
        if time_budget_s is not None and spent > time_budget_s:
            raise TimeoutError(f"{session.name}: the policy used more than {time_budget_s:.0f} s of compute")
        u = sim.step(action)
        obs = state.apply(u)
    res = sim.result()
    res.compute_s = spent
    return res


def validate(policy_dir="my_policy.py", sessions: Optional[Sequence[Session]] = None, *, only: Optional[Sequence[str]] = None,
             verbose: bool = True) -> dict:
    """Leave-one-session-out: for each session, train on the others, replay it.

    This is how grading works too: there, you train on all four sessions and the policy
    plays two sessions it has never seen. As in grading, the session played is presented as
    a real day (``Session.as_day``: rows in no particular order, ids re-issued); the
    training sessions are as recorded. Returns {session: EpisodeResult}."""
    module = load_policy_module(policy_dir)
    sessions = list(sessions) if sessions is not None else training_sessions()
    out = {}
    for held in sessions:
        if only and held.name not in only:
            continue
        train = [s for s in sessions if s.name != held.name]
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            module.train(train, Path(tmp))
            t_train = time.perf_counter() - t0
            policy = module.load(Path(tmp))
            res = run_policy(policy, held.as_day(), fit_clock(train, held.presets))
        out[held.name] = res
        if verbose:
            print(f"  {held.name:10s} {res.hours:7.2f} h   {res.micrographs:6d} photos   "
                  f"{res.square_changes:3d} square changes   train {t_train:5.1f} s, play {res.compute_s:5.1f} s",
                  flush=True)
    return out


def train_policy(policy_dir, out_dir=None, sessions: Optional[Sequence[Session]] = None) -> Path:
    """Train on every training session and write the artefacts you will submit."""
    module = load_policy_module(policy_dir)
    out_dir = Path(out_dir) if out_dir else artifacts_dir(policy_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    module.train(list(sessions) if sessions is not None else training_sessions(), out_dir)
    return out_dir
