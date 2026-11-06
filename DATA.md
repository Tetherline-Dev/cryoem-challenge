# The data and the simulator, in detail

The reference for when you need it. `README.md` and the notebook come first.

## Sessions

| session | specimen | squares | holes | photos | particles the expert kept | recorded length |
|---|---|---|---|---|---|---|
| grid1 | apoferritin | 3 | 371 | 2,597 | 66,668 | 3.0 h |
| codh | CODH/ACS | 9 | 2,779 | 13,890 | 75,311 | 18.3 h |
| delta10 | E. coli 70S ribosome | 12 | 1,214 | 8,486 | 338,642 | 40.0 h |
| deltaRlmE | E. coli 70S ribosome | 51 | 1,468 | 10,254 | 336,173 | 20.1 h |

All four are public EMPIAR datasets. The expert's selection of particles is unpublished
and shared for this challenge only. In code, `challenge.training_sessions()` returns
them, and `challenge.load_session("data/grid1")` returns one.

## Files in `data/<session>/`

**`holes.parquet`**: one row per hole the operator selected, in the order they were
first shot.

| column | meaning |
|---|---|
| `hole_id`, `square_id` | identifiers |
| `sort_key` | a stable string order (the references use it to break ties) |
| `x_px`, `y_px`, `diameter_px` | the hole's centre and size in its square's overview image (full-size pixels) |
| `n_micrographs` | photos a `ShootHole` takes there (3 to 7) |
| `has_embedding`, `has_image` | whether the hole has an image vector and a crop |
| `px_norm`, `py_norm` | position in the square image, as a fraction of its width and height |
| `dist_center`, `dist_edge` | distance to the square's centre and to its nearest edge |
| `log_intensity`, `log_intensity_std` | brightness of the hole in the overview, and its spread, standardised per session. Thicker ice is darker. |
| `intensity_z_square`, `intensity_rank_square` | brightness relative to the other holes of its square |
| `nbr_log_intensity`, `nbr_intensity_contrast` | mean brightness of the six nearest holes, and the hole's contrast to them |
| `near_grid_bar` | the hole sits next to a bar of the metal grid |
| `order_norm` | the order in which the microscope's software would visit the holes of the square |
| `geometry_known` | the square's geometry was available to that software |

**`squares.parquet`**: one row per square.

| column | meaning |
|---|---|
| `square_id`, `n_holes` | identifier, and selected holes in it |
| `n_targets`, `log_target_count` | all the holes the software found in the square |
| `all_log_intensity_mean`, `near_grid_bar_fraction` | the square's mean hole brightness, and its share of holes near a grid bar |
| `stage_x_um`, `stage_y_um` | where the square sits on the grid (µm) |
| `centroid_x_um`, `centroid_y_um` | the mean stage position of the square's holes (missing for codh) |
| `overview_width_px`, `overview_height_px` | size of the full overview image |

**`micrographs.parquet`**: one row per photo. Training sessions only; a policy never
sees this file.

| column | meaning |
|---|---|
| `micrograph_id`, `hole_id`, `square_id`, `shot_index`, `timestamp_s` | which photo, and when it was recorded |
| `picks`, `ctf_fit_A`, `defocus_um`, `ice_rel`, `motion_total_px`, `motion_early_px` | the software's report on the photo (CryoSPARC Live): particles it picked, the resolution in Å to which the photo's signal fits (lower is sharper), measured focus, relative ice thickness, sample motion. Your policy receives these 75 s after the photo. |
| `curated_count` | **the score**: the photo's particles the expert kept |
| `total_count` | the photo's particles that survived the expert's first, looser cleaning (an extra label) |

**`embeddings.npy`**: a 768-number vector per hole, rows aligned with `holes.parquet`.
It comes from a pretrained cryo-EM image encoder (Cryo-IEF) applied to the hole's crop.
Rows are zero where `has_embedding` is false.

**`hole_images.npy`**: a 64×64 crop of the square overview around every hole, two hole
diameters wide. It is one uint8 array, rows aligned with `holes.parquet`, all zeros where
`has_image` is false. Read all of them with `session.hole_images`, one with
`session.hole_image(hole_id)`.

**`square_images.parquet`**: each square's overview as JPEG bytes, long side 1024 px. Read
one with `session.square_image(square_id)`.

Contrast is set once per square, so brightness compares between the holes of a square.
These overviews are all anyone sees of a hole before it is shot.

**`session.json`**: specimen, optics, counts, and the session's own clock.

On a hidden session your policy gets the same files except `micrographs.parquet` and the
clock.

## The simulator

* **Actions.** `GoToSquare(square_id)` moves to a square. You must choose one whenever
  `obs.current_square` is None: at the start, and after a square runs out of selected
  holes. `ShootHole(hole_id)` takes every photo of a hole in the current square, and the
  hole is then spent.
* **Timings.** Each session's clock is fitted from its own recorded time stamps:

  | action | duration |
  |---|---|
  | one photo | 3 to 13 s |
  | move to a near hole | 0 to 9 s |
  | move to a far hole, where the session shows a difference | 37 to 45 s |
  | square change | 80 to 180 s |

  The simulator charges the session's own clock. Your policy sees `obs.clock`, an
  estimate that starts from the training sessions' clock and learns from the day's moves.
* **Reports.** Each photo's report arrives 75 instrument-seconds after it is taken, in
  `obs.new_results`. `obs.pending` counts the photos still waiting for theirs.
* **Stop.** The session ends as soon as the photos taken hold 75% of the particles the
  expert kept from the whole session. The score is the instrument hours it took.

`obs` also offers:

* `obs.holes_left()` and `obs.squares_left()`
* `obs.remaining`, a boolean array aligned with `info.holes`
* `obs.cost_s(hole_id)`, the seconds to shoot a hole from where you are
* `obs.distance_px(a, b)`

`challenge/policy.py` documents all of it.

## The references

| reference | what it does |
|---|---|
| random | picks a random square, then random holes in it |
| raster | always takes the nearest hole not yet shot, then the nearest square; blind to quality |
| operator | the session's own recorded order, replayed |
| oracle | knows every hole's outcome in advance and walks a short path through the best ones |

The oracle is a strong reference, not a proven optimum. Random and the raster depend on
a random start and are averaged over two seeds. Grading also shows two frontier language
models that played the same sessions through an interactive version of the simulator.
