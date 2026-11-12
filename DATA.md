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

**`holes.parquet`**: one row per hole the operator selected. On the training sessions the
rows are in the order the holes were first shot; a hidden session's come in no particular
order, with its ids re-issued (and so does the session `validate` plays).

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
| `order_norm` | the hole's place in the order EPU's hole finder numbered the holes of its square, which EPU's default path follows in part |
| `geometry_known` | the square's geometry was available to that software |

**`squares.parquet`**: one row per square.

| column | meaning |
|---|---|
| `square_id`, `n_holes` | identifier, and selected holes in it |
| `n_targets`, `log_target_count` | all the holes the software found in the square |
| `all_log_intensity_mean`, `near_grid_bar_fraction` | the square's mean hole brightness, and its share of holes near a grid bar |
| `stage_x_um`, `stage_y_um` | where the square sits on the grid (µm); square changes are priced by this distance |
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

**`session.json`**: specimen, optics, counts, and:

* `epu`: the session's EPU settings, fixed before the first exposure: `clustering_mode` and
  `clustering_radius_um` (holes within this distance of where the stage is parked are reached
  by image shift), `autofocus_recurrence` and `autofocus_distance_um` (an autofocus after every
  stage move, or once the stage is that far from the last one), and `square_pixel_nm` (the
  pixel size of the square overview, so `x_px` × `square_pixel_nm` / 1000 is in µm). Also
  `info.presets`.
* `clock`: the session's own clock (below).
* `square_changes`: the session's recorded square changes in seconds, and how far apart the
  squares were (mm), which `fit_clock` pools.

On a hidden session your policy gets the same files except `micrographs.parquet`, the clock
and the recorded square changes. It does get `epu`.

## The simulator

* **Actions.** `GoToSquare(square_id)` moves to a square. You must choose one whenever
  `obs.current_square` is None: at the start, and after a square runs out of selected
  holes. `ShootHole(hole_id)` takes every photo of a hole in the current square, and the
  hole is then spent.
* **Timings.** Moves cost what EPU spends. EPU parks the stage at a hole, the **anchor**,
  and reaches every hole within the clustering radius of it by image shift. A hole beyond
  the radius needs a **stage move**: the stage travels, settles and re-centres, and
  autofocuses when the session's autofocus rule says one is due; that hole becomes the new
  anchor. A square change includes centring on the new square's first hole, which becomes
  the anchor. Each session's clock is the mean of its own recorded transitions of each kind,
  operator breaks left out:

  | action | grid1 | codh | delta10 | deltaRlmE |
  |---|---|---|---|---|
  | one photo | 3.9 s | 3.4 s | 11.7 s | 4.1 s |
  | hop: a hole within the radius of the anchor | 0.1 s | 0.1 s | 9.8 s (a short stage hop) | 0.9 s |
  | stage move: any other hole | 53 s | 76 s | 55 s | 45 s |
  | autofocus on top, when due | after every stage move | after every stage move | +20 s, once 10 µm from the last | +7.5 s, once 20 µm from the last |
  | square change | 116 s | 120 s, +31 s per mm farther | 174 s | 99 s |
  | clustering radius | 12 µm | 6 µm | 6 µm | 8 µm |

  A stage move also grows slightly with its distance from the anchor on codh, delta10 and
  deltaRlmE (0.1 to 0.7 s per µm). Priced this way, the operator's own recorded order takes
  what the session really took, breaks apart, to within 0-4%. `challenge/clock.py` has the
  details. The simulator charges the session's own clock. Your policy sees `obs.clock`, an
  estimate that starts from the training sessions' clock under the session's own EPU
  settings and learns from the day's moves.
* **Reports.** Each photo's report arrives 75 instrument-seconds after it is taken, in
  `obs.new_results`. `obs.pending` counts the photos still waiting for theirs.
* **Stop.** The session ends as soon as the photos taken hold 75% of the particles the
  expert kept from the whole session. The score is the instrument hours it took.

`obs` also offers:

* `obs.holes_left()` and `obs.squares_left()`
* `obs.remaining`, a boolean array aligned with `info.holes`
* `obs.anchor_hole`, where the stage is parked, and `obs.autofocus_hole`
* `obs.move_s(hole_id)` and `obs.move_kind(hole_id)` ('hop', 'stage', 'stage_af', or 'entry'
  for a square's first hole), priced from the anchor by your clock estimate
* `obs.cost_s(hole_id)`, the seconds to shoot a hole from where you are
* `obs.square_change_s(square_id)`, the seconds to move there from the square the stage is in
* `obs.distance_px(a, b)` and `obs.distance_um(a, b)`

`challenge/policy.py` documents all of it.

## The references

| reference | what it does |
|---|---|
| random | picks a random square, then random holes in it |
| raster | always takes the nearest hole not yet shot, then the nearest square; blind to quality and to where the stage is parked |
| EPU raster | what EPU does on its own: the raster's squares, and in each square EPU's clusters (park at the hole whose radius covers the most holes left, shoot them all, repeat); blind to quality |
| operator | the session's own recorded order, replayed |
| oracle | knows every hole's outcome in advance and visits the best ones cluster by cluster |

The oracle is a strong reference, not a proven optimum. Random and the two rasters depend
on a random first square and are averaged over ten. Grading also shows two frontier
language models that played the same sessions through an interactive version of the
simulator, re-priced on this clock.

**Grading.** Grading plays each hidden session as a real day would present it: the rows of
every table in no particular order and every id re-issued. Positions, images, EPU's
settings and `order_norm` are as described above. `validate` plays the held-out training
session the same way (`Session.as_day()`), and the references above are scored on that
view.
