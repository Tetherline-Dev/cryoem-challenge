#!/usr/bin/env python3
"""The challenge from the command line (the notebook does the same).

    python main.py download                          # once: the four training sessions into data/
    python main.py validate                         # my_policy.py: train on three sessions, play the fourth, in turn
    python main.py validate --policy bayes_policy.py # the other baseline
    python main.py submit                            # train my_policy.py on all four, check it, write submission.zip
    python main.py train                             # just the training (-> artifacts/)
    python main.py references                        # recompute data/references.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from challenge import (fit_clock, format_scorecard, load_references, reference_hours, scorecard,  # noqa: E402
                       training_sessions, validate)
from challenge.data import download_data  # noqa: E402
from challenge.runner import train_policy  # noqa: E402
from challenge.scoring import REFERENCES_FILE  # noqa: E402
from challenge.submit import check_and_pack  # noqa: E402


def cmd_validate(a) -> None:
    only = a.sessions.split(",") if a.sessions else None
    print(f"leave-one-session-out: {a.policy}")
    res = validate(a.policy, only=only)
    refs = load_references()
    if refs:
        print()
        print(format_scorecard(scorecard({Path(a.policy).stem: {s: r.hours for s, r in res.items()}}, refs)))


def cmd_train(a) -> None:
    print(f"trained on every training session; artefacts in {train_policy(a.policy)}")


def cmd_submit(a) -> None:
    check_and_pack(a.policy)


def cmd_references(a) -> None:
    sessions = training_sessions()
    out = {}
    for s in sessions:
        out[s.name] = reference_hours(s, fit_clock([o for o in sessions if o.name != s.name]))
        print(f"  {s.name:10s} " + "  ".join(f"{k} {v:.2f} h" for k, v in out[s.name].items()), flush=True)
    REFERENCES_FILE.write_text(json.dumps(out, indent=1))
    print(f"wrote {REFERENCES_FILE}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("validate", cmd_validate), ("train", cmd_train), ("submit", cmd_submit)):
        p = sub.add_parser(name)
        p.add_argument("--policy", default=str(ROOT / "my_policy.py"), help="policy file (default: my_policy.py)")
        if name == "validate":
            p.add_argument("--sessions", default=None, help="hold out only these (comma-separated)")
        p.set_defaults(fn=fn)
    sub.add_parser("references").set_defaults(fn=cmd_references)
    sub.add_parser("download").set_defaults(fn=lambda a: download_data())
    a = ap.parse_args()
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
