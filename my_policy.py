"""The starting policy, and the one file you edit.

A small neural network scores every hole; a simple rule turns the scores into moves.
It decides in four steps, one section each below:

  inputs    describe every hole by what is known before it is shot
  network   five small networks predict each hole's curated particles per photo
  day       the reports on photos already taken shift every prediction up or down
  decision  shoot the hole with the most predicted particles per second of instrument
            time, or move to another square when one is worth the trip

Grading calls only `load`. The contract:

  train(sessions, out_dir)   fit on labelled sessions; save what you need in out_dir
  load(out_dir) -> Policy    rebuild the policy from out_dir

  python main.py validate    train on three sessions, play the fourth, in turn
  python main.py submit      train on all four, check, write submission.zip

The notebook walks through each section. If you edit this file in Colab through the
notebook's cells, run save_policy() there to write them back here.
"""

# %% setup
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn

from challenge import GoToSquare, HOLE_FEATURES, LIVE_COLUMNS, Policy, ShootHole


# %% inputs: what is known about a hole before it is shot
N_COMPONENTS = 8   # principal components of the 768-number image vector to keep


def fit_inputs(sessions):
    """Fit the recipe that turns a session's holes into a table of numbers: a projection
    of the image vectors, then a mean and a scale per column, all from the training holes."""
    emb = np.vstack([s.embeddings[s.holes["has_embedding"].to_numpy()] for s in sessions])
    centre = emb.mean(axis=0)
    _, _, vt = np.linalg.svd(emb - centre, full_matrices=False)
    recipe = {"centre": centre, "components": vt[:N_COMPONENTS]}
    raw = np.vstack([raw_inputs(s, recipe) for s in sessions])
    sd = raw.std(axis=0)
    recipe["mean"] = raw.mean(axis=0)
    recipe["scale"] = np.where(sd > 1e-3, sd, 1.0)    # a column constant in training stays unscaled
    return recipe


def raw_inputs(session, recipe):
    feats = session.holes[list(HOLE_FEATURES)].to_numpy(dtype=float)
    pcs = (session.embeddings - recipe["centre"]) @ recipe["components"].T
    pcs[~session.holes["has_embedding"].to_numpy()] = 0.0
    return np.nan_to_num(np.hstack([feats, pcs]))


def hole_inputs(session, recipe):
    """One row per hole (session.holes order): the 13 hole features and 8 image components,
    standardised and clipped at 5 standard deviations (a new session can sit far outside the
    training range on some column)."""
    return np.clip((raw_inputs(session, recipe) - recipe["mean"]) / recipe["scale"], -5.0, 5.0)


def hole_targets(sessions):
    """What the networks learn: each hole's mean curated particles per photo, as a z-score
    within its session (sessions differ a lot in scale), and the pooled scale that turns a
    z-score back into particles."""
    zs, means, sds, within = [], [], [], []
    for s in sessions:
        y = s.hole_yield().to_numpy(dtype=float)
        zs.append((y - y.mean()) / (y.std() + 1e-9))
        means.append(y.mean())
        sds.append(y.std())
        m = s.micrographs
        within.append(float((m["curated_count"] - m.groupby("hole_id")["curated_count"].transform("mean")).std()))
    scale = {"mean": float(np.mean(means)),          # an average hole's particles per photo
             "between_sd": float(np.mean(sds)),      # how much holes differ
             "within_sd": float(np.mean(within))}    # how much photos of one hole differ
    return np.concatenate(zs), scale


# %% network: predict each hole's yield from its inputs
N_NETS, HIDDEN, EPOCHS = 5, 64, 300


def make_net(n_inputs):
    return nn.Sequential(nn.Linear(n_inputs, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, 1))


def fit_nets(x, z):
    """Five small networks trained on the same data from different seeds. Where they
    disagree, the prediction is uncertain."""
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(z, dtype=torch.float32).unsqueeze(1)
    nets = []
    for k in range(N_NETS):
        torch.manual_seed(k)
        net = make_net(x.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(EPOCHS):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(x), y)
            loss.backward()
            opt.step()
        nets.append(net.eval())
    return nets


def predict(nets, x):
    """Mean and spread of the five predictions, in z-units."""
    x = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        p = np.stack([net(x).squeeze(1).numpy() for net in nets])
    return p.mean(axis=0), p.std(axis=0)


# %% day: what the reports on photos already taken say
def report_inputs(rows):
    """rows: one per photo, the report's fields in LIVE_COLUMNS order."""
    picks, ctf, defocus, ice, motion, motion_early = (rows[:, i] for i in range(6))
    ctf_ok = np.isfinite(ctf) & (ctf > 0) & (ctf <= 20)
    ctf = np.where(ctf_ok, ctf, 20.0)
    return np.nan_to_num(np.column_stack([
        np.log1p(np.clip(picks, 0, None)), ctf, ctf_ok.astype(float), defocus, ice,
        np.log1p(np.abs(motion)), np.log1p(np.abs(motion_early)), np.ones(len(rows))]))


def fit_report_model(sessions, alpha=1.0):
    """A ridge regression from a photo's report to log(1 + its curated count)."""
    rows = np.vstack([s.micrographs[list(LIVE_COLUMNS)].to_numpy(dtype=float) for s in sessions])
    y = np.concatenate([np.log1p(s.micrographs["curated_count"].to_numpy(dtype=float)) for s in sessions])
    x = report_inputs(rows)
    return np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)


def guess_particles(weights, results):
    """The curated particles each reported photo probably holds."""
    rows = np.array([[getattr(r, c) for c in LIVE_COLUMNS] for r in results], dtype=float)
    return np.clip(np.expm1(report_inputs(rows) @ weights), 0.0, None)


# %% decision: from predictions to one move
BETA = 1.0     # optimism: how much a hole's uncertainty adds to its value
STAY = 12.0    # photos over which a square change is spread when comparing squares


class Grid:
    """The session's holes as arrays, built once at reset."""

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
    """mean, sd: predicted curated particles per photo of every hole (info.holes order)."""
    clock = obs.clock
    value = mean + BETA * sd
    left = obs.remaining
    # Another square: each of its holes valued per second as if we moved there now, with the
    # square change spread over the photos we expect to take there. A square is worth its
    # best hole.
    photos_left = np.zeros(len(grid.square_ids))
    np.add.at(photos_left, grid.sq, np.where(left, grid.n, 0.0))
    stay = np.clip(np.minimum(STAY, photos_left), 1.0, None)
    seconds_after_move = (clock.near_move_s + grid.n * clock.shot_s) / grid.n + clock.square_change_s / stay[grid.sq]
    square_rate = np.full(len(grid.square_ids), -np.inf)
    np.maximum.at(square_rate, grid.sq, np.where(left, value / seconds_after_move, -np.inf))
    if obs.current_square is None:
        return GoToSquare(grid.square_ids[int(np.argmax(square_rate))])
    here = grid.square_ids.index(obs.current_square)
    just_arrived = obs.beam_hole is None          # a square change was just paid for: shoot here first
    rows = np.flatnonzero((grid.sq == here) & left)
    # This square: seconds per photo of each hole, counting the move from where the beam is.
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


# %% policy: the pieces put together
def train(sessions, out_dir):
    recipe = fit_inputs(sessions)
    z, scale = hole_targets(sessions)
    nets = fit_nets(np.vstack([hole_inputs(s, recipe) for s in sessions]), z)
    saved = {"recipe": recipe, "scale": scale, "nets": [net.state_dict() for net in nets],
             "report_model": fit_report_model(sessions)}
    with open(Path(out_dir) / "policy.pkl", "wb") as f:
        pickle.dump(saved, f)


def load(out_dir):
    with open(Path(out_dir) / "policy.pkl", "rb") as f:
        saved = pickle.load(f)
    nets = []
    for state in saved["nets"]:
        net = make_net(len(saved["recipe"]["mean"]))
        net.load_state_dict(state)
        nets.append(net.eval())
    return MyPolicy(saved["recipe"], saved["scale"], nets, saved["report_model"])


class MyPolicy(Policy):
    def __init__(self, recipe, scale, nets, report_model):
        self.recipe, self.scale, self.nets, self.report_model = recipe, scale, nets, report_model

    def reset(self, info):
        """A new session: predict every hole before anything is shot."""
        self.grid = Grid(info)
        z, z_sd = predict(self.nets, hole_inputs(info, self.recipe))
        self.prior_mean = self.scale["mean"] + self.scale["between_sd"] * z
        self.prior_sd = self.scale["between_sd"] * z_sd + 1e-6
        self.gap_sum, self.gap_n = 0.0, 0       # reported minus predicted, over the day's photos

    def act(self, obs):
        if obs.new_results:
            seen = guess_particles(self.report_model, obs.new_results)
            rows = [self.grid.row[r.hole_id] for r in obs.new_results]
            self.gap_sum += float(np.sum(seen - self.prior_mean[rows]))
            self.gap_n += len(rows)
        # shift every prediction by the day's average gap, trusted more as photos come in
        precision = self.gap_n / self.scale["within_sd"] ** 2
        trust = precision / (precision + 1.0 / self.scale["between_sd"] ** 2)
        shift = trust * (self.gap_sum / self.gap_n if self.gap_n else 0.0)
        return decide(obs, self.grid, self.prior_mean + shift, self.prior_sd)
