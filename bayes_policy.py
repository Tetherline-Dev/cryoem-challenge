"""A second baseline: a Bayesian linear model instead of a neural network.

  inputs    the same as my_policy.py: 13 hole features and 8 image components
  prior     a Bayesian ridge regression (scikit-learn) predicts each hole's curated
            particles per photo, with an uncertainty
  day       within each square, the gap between what its photos reported and what was
            predicted shifts the square's remaining holes, by a weight that grows with
            the evidence (a conjugate normal update), and shrinks their uncertainty
  decision  the same rule as my_policy.py: particles per second, or move square

    python main.py validate --policy bayes_policy.py

Self-contained on purpose: it shares no code with my_policy.py, so editing one never
breaks the other.
"""
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import BayesianRidge

from challenge import GoToSquare, HOLE_FEATURES, LIVE_COLUMNS, Policy, ShootHole

N_COMPONENTS, BETA, STAY = 8, 1.0, 12.0


# ----- inputs and targets (as in my_policy.py) -----------------------------------------
def fit_inputs(sessions):
    emb = np.vstack([s.embeddings[s.holes["has_embedding"].to_numpy()] for s in sessions])
    centre = emb.mean(axis=0)
    _, _, vt = np.linalg.svd(emb - centre, full_matrices=False)
    recipe = {"centre": centre, "components": vt[:N_COMPONENTS]}
    raw = np.vstack([raw_inputs(s, recipe) for s in sessions])
    sd = raw.std(axis=0)
    recipe["mean"], recipe["scale"] = raw.mean(axis=0), np.where(sd > 1e-3, sd, 1.0)
    return recipe


def raw_inputs(session, recipe):
    feats = session.holes[list(HOLE_FEATURES)].to_numpy(dtype=float)
    pcs = (session.embeddings - recipe["centre"]) @ recipe["components"].T
    pcs[~session.holes["has_embedding"].to_numpy()] = 0.0
    return np.nan_to_num(np.hstack([feats, pcs]))


def hole_inputs(session, recipe):
    return np.clip((raw_inputs(session, recipe) - recipe["mean"]) / recipe["scale"], -5.0, 5.0)


def hole_targets(sessions):
    zs, means, sds, within = [], [], [], []
    for s in sessions:
        y = s.hole_yield().to_numpy(dtype=float)
        zs.append((y - y.mean()) / (y.std() + 1e-9))
        means.append(y.mean())
        sds.append(y.std())
        m = s.micrographs
        within.append(float((m["curated_count"] - m.groupby("hole_id")["curated_count"].transform("mean")).std()))
    return np.concatenate(zs), {"mean": float(np.mean(means)), "between_sd": float(np.mean(sds)),
                                "within_sd": float(np.mean(within))}


# ----- the day's reports (as in my_policy.py) ------------------------------------------
def report_inputs(rows):
    picks, ctf, defocus, ice, motion, motion_early = (rows[:, i] for i in range(6))
    ctf_ok = np.isfinite(ctf) & (ctf > 0) & (ctf <= 20)
    ctf = np.where(ctf_ok, ctf, 20.0)
    return np.nan_to_num(np.column_stack([
        np.log1p(np.clip(picks, 0, None)), ctf, ctf_ok.astype(float), defocus, ice,
        np.log1p(np.abs(motion)), np.log1p(np.abs(motion_early)), np.ones(len(rows))]))


def fit_report_model(sessions, alpha=1.0):
    rows = np.vstack([s.micrographs[list(LIVE_COLUMNS)].to_numpy(dtype=float) for s in sessions])
    y = np.concatenate([np.log1p(s.micrographs["curated_count"].to_numpy(dtype=float)) for s in sessions])
    x = report_inputs(rows)
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def guess_particles(weights, results):
    rows = np.array([[getattr(r, c) for c in LIVE_COLUMNS] for r in results], dtype=float)
    return np.clip(np.expm1(report_inputs(rows) @ weights), 0.0, None)


# ----- the decision (as in my_policy.py) ------------------------------------------------
class Grid:
    def __init__(self, info):
        h = info.holes
        self.hole_ids = h["hole_id"].tolist()
        self.square_ids = info.squares["square_id"].tolist()
        index = {s: i for i, s in enumerate(self.square_ids)}
        self.sq = np.array([index[s] for s in h["square_id"]])
        self.n = h["n_micrographs"].to_numpy(dtype=float)
        self.x = h["x_px"].to_numpy(dtype=float)
        self.y = h["y_px"].to_numpy(dtype=float)
        self.row = info.hole_row


def decide(obs, grid, mean, sd):
    clock = obs.clock
    value = mean + BETA * sd
    left = obs.remaining
    photos_left = np.zeros(len(grid.square_ids))
    np.add.at(photos_left, grid.sq, np.where(left, grid.n, 0.0))
    stay = np.clip(np.minimum(STAY, photos_left), 1.0, None)
    seconds_after_move = (clock.near_move_s + grid.n * clock.shot_s) / grid.n + clock.square_change_s / stay[grid.sq]
    square_rate = np.full(len(grid.square_ids), -np.inf)
    np.maximum.at(square_rate, grid.sq, np.where(left, value / seconds_after_move, -np.inf))
    if obs.current_square is None:
        return GoToSquare(grid.square_ids[int(np.argmax(square_rate))])
    here = grid.square_ids.index(obs.current_square)
    just_arrived = obs.beam_hole is None
    rows = np.flatnonzero((grid.sq == here) & left)
    move = np.full(len(rows), clock.near_move_s)
    if obs.beam_hole is not None and clock.far_move_s is not None:
        b = grid.row[obs.beam_hole]
        d = np.hypot(grid.x[rows] - grid.x[b], grid.y[rows] - grid.y[b])
        move = np.where(d >= clock.far_threshold_px, clock.far_move_s, clock.near_move_s)
    rates = value[rows] / ((move + grid.n[rows] * clock.shot_s) / grid.n[rows])
    best = int(np.argmax(rates))
    square_rate[here] = -np.inf
    if not just_arrived and np.max(square_rate) > rates[best]:
        return GoToSquare(grid.square_ids[int(np.argmax(square_rate))])
    return ShootHole(grid.hole_ids[rows[best]])


# ----- the policy ------------------------------------------------------------------------
def train(sessions, out_dir):
    recipe = fit_inputs(sessions)
    z, scale = hole_targets(sessions)
    model = BayesianRidge().fit(np.vstack([hole_inputs(s, recipe) for s in sessions]), z)
    with open(Path(out_dir) / "policy.pkl", "wb") as f:
        pickle.dump({"recipe": recipe, "scale": scale, "model": model,
                     "report_model": fit_report_model(sessions)}, f)


def load(out_dir):
    with open(Path(out_dir) / "policy.pkl", "rb") as f:
        return BayesPolicy(**pickle.load(f))


class BayesPolicy(Policy):
    def __init__(self, recipe, scale, model, report_model):
        self.recipe, self.scale, self.model, self.report_model = recipe, scale, model, report_model

    def reset(self, info):
        self.grid = Grid(info)
        z, z_sd = self.model.predict(hole_inputs(info, self.recipe), return_std=True)
        self.prior_mean = self.scale["mean"] + self.scale["between_sd"] * z
        self.prior_sd = self.scale["between_sd"] * z_sd
        n_squares = len(self.grid.square_ids)
        self.gap_sum = np.zeros(n_squares)      # reported minus predicted, per square
        self.gap_n = np.zeros(n_squares)

    def act(self, obs):
        if obs.new_results:
            seen = guess_particles(self.report_model, obs.new_results)
            for r, y in zip(obs.new_results, seen):
                i = self.grid.row[r.hole_id]
                self.gap_sum[self.grid.sq[i]] += y - self.prior_mean[i]
                self.gap_n[self.grid.sq[i]] += 1
        precision = self.gap_n / self.scale["within_sd"] ** 2
        trust = precision / (precision + 1.0 / self.scale["between_sd"] ** 2)
        shift = trust * np.divide(self.gap_sum, self.gap_n, out=np.zeros_like(self.gap_sum), where=self.gap_n > 0)
        mean = self.prior_mean + shift[self.grid.sq]
        sd = self.prior_sd * np.sqrt(1.0 - trust[self.grid.sq])
        return decide(obs, self.grid, mean, sd)
