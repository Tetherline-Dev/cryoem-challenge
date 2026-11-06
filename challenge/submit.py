"""Checking your policy and packing it for submission.

    python main.py submit        # trains my_policy.py on all four sessions, checks it, writes submission.zip

The zip holds my_policy.py, artifacts/ (what train saved), extra_requirements.txt if you
need pip packages beyond requirements.txt, and any other file or folder you added at the
top of the challenge folder (helper modules, weights). Grading unpacks it next to its own
copy of challenge/, installs the extra packages, calls ``load(artifacts)`` with no network
access, and plays the hidden sessions under the limits below.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

from .clock import fit_clock
from .data import PACKAGE, training_sessions
from .runner import load_policy_module, run_policy, train_policy

MAX_SUBMISSION_BYTES = 1_000_000_000      # code + artefacts, pretrained weights included
COMPUTE_BUDGET_S = 15 * 60                # the policy's own compute, per hidden session
# what we shipped: not part of your submission
SHIPPED = {"README.md", "DATA.md", "START_HERE.ipynb", "main.py", "requirements.txt", "bayes_policy.py",
           "challenge", "data", "figures", "submission.zip", ".ipynb_checkpoints", "__pycache__", ".DS_Store",
           ".git", ".config", "sample_data"}
SKIP_PARTS = ("__pycache__", ".ipynb_checkpoints", ".DS_Store")


def submission_files(root: Path) -> list:
    files = []
    for p in sorted(root.iterdir()):
        if p.name in SHIPPED or p.name.startswith("artifacts_"):
            continue
        items = [p] if p.is_file() else [q for q in p.rglob("*") if q.is_file()]
        files += [q for q in items if not any(s in q.parts for s in SKIP_PARTS)]
    return files


def check_and_pack(policy_file="my_policy.py", out_zip=None, *, train: bool = True) -> Path:
    policy_file = Path(policy_file).resolve()
    root = policy_file.parent
    if policy_file.name != "my_policy.py":
        raise SystemExit("submit packs my_policy.py: put your policy there")
    sessions = training_sessions()
    if train:
        print("training my_policy.py on all four sessions ...", flush=True)
        train_policy(policy_file, sessions=sessions)
    artifacts = root / "artifacts"
    if not artifacts.is_dir():
        raise SystemExit("no artifacts/: run `python main.py train` first")
    module = load_policy_module(policy_file)
    smoke = min(sessions, key=lambda s: len(s.holes))
    print(f"checking: load(artifacts) and a full replay of {smoke.name} (in-sample, so its hours mean nothing)")
    t0 = time.perf_counter()
    res = run_policy(module.load(artifacts), smoke, fit_clock(sessions), time_budget_s=COMPUTE_BUDGET_S)
    print(f"  ok: {res.hours:.2f} h, {res.micrographs} photos, load + play {time.perf_counter() - t0:.1f} s")
    per_hole = res.compute_s / max(res.holes, 1)
    if per_hole * 4000 > COMPUTE_BUDGET_S:
        print(f"  warning: {per_hole * 1000:.0f} ms per decision; a hidden session can have ~4,000 holes "
              f"and the budget is {COMPUTE_BUDGET_S // 60} min of compute per session")
    files = submission_files(root)
    size = sum(p.stat().st_size for p in files)
    if size > MAX_SUBMISSION_BYTES:
        raise SystemExit(f"the submission is {size / 1e6:.0f} MB; the limit is {MAX_SUBMISSION_BYTES / 1e6:.0f} MB")
    out_zip = Path(out_zip) if out_zip else root / "submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("submission") / p.relative_to(root))
    print(f"wrote {out_zip} ({size / 1e6:.1f} MB): " + ", ".join(sorted({str(p.relative_to(root)).split('/')[0] for p in files})))
    return out_zip
