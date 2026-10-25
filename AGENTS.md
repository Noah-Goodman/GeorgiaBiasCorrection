# AGENTS.md

Bias correction for six low-cost QuantAQ MODULAIR sensors. All six were colocated with high-quality EPA (FRM/FEM)
monitors for six weeks, May 31 – July 16, 2025. The colocation data trains one ML model per sensor per species, mapping
the sensor's raw readings to the EPA reference. The models then correct each sensor's current data from wherever it
is deployed now.

## Layout

Each sensor has its own top-level folder named after where it is now. Each folder holds a full copy of the pipeline.
Three of the six sensors are in the repo so far:

| Code folder    | Data folder         | Sensor    |
| -------------- | ------------------- | --------- |
| `Chapel_Hill/` | `.data/Chapel-Hill/` | MOD-00592 |
| `Lithonia/`    | `.data/Lithonia/`    | MOD-00595 |
| `Pittsburgh/`  | `.data/Pittsburgh/`  | MOD-00591 |

The folder names differ: code folders use an underscore and data folders use a hyphen (`Chapel_Hill` vs `Chapel-Hill`).

Each location folder contains:

- `preprocess.py`: builds hourly features from the colocation data, aligns them to GAPA, fits the rh/temp scaler and
  writes the 80/20 train/test split
- `modelBakeoff.py`: tunes and compares candidate models (see [Model bakeoff](#model-bakeoff))
- `train.py`: trains the chosen model for a species and saves it
- `preprocessCurrent.py`: builds the same features from current data, using the saved scaler
- `visualization.py`: plots the test split (`-f train`) or corrected current data (`-f current`)
- `models/[species]-model.pkl`, `scalers/rh-temp-scaler-[species].pkl`, `figures/*.png`: outputs, committed to git

The scripts are near-copies of each other. Data paths are hardcoded as `../.data/<Location>/...`, and outputs go to
`models/`, `scalers/` and `figures/`. So **run every script from inside its location folder**:
`cd Pittsburgh && python preprocess.py no2`.

Species: `pm25`, `pm10`, `no`, `no2`, `o3`, `co`. PM is in µg/m³. Gases are in ppb.

## Data (`.data/`)

All data is local and is never committed. `.data/` and `.venv/` are gitignored. Each contributor downloads the data
themselves. Downloading is manual for now, and automating it is planned. Don't commit anything from `.data/`, and don't
move data out of it into tracked folders.

```
.data/
└── [Location]/
    ├── collocation/                        # colocation period, May 31 – July 16, 2025
    │   ├── raw.csv                         # QuantAQ export for this sensor (downloaded)
    │   ├── gapa-[species].csv              # GAPA reference data for one species (downloaded)
    │   ├── preprocessed-[species].csv      # hourly features indexed by dayhour (preprocess.py)
    │   └── gapa-preprocessed-[species].csv # GAPA target aligned to those dayhours (preprocess.py)
    ├── current/                            # sensor at its current location
    │   ├── raw.csv                         # QuantAQ export for this sensor (downloaded)
    │   └── preprocessed-[species].csv      # hourly features indexed by dayhour (preprocessCurrent.py)
    ├── hyperParams/
    │   ├── [species].json                  # best params per model and search (modelBakeoff.py)
    │   └── [species]-scores.csv            # test scores per model and search (modelBakeoff.py)
    └── split/                              # 80/20 holdout, seed 10 (preprocess.py)
        ├── xTrain-[species].csv
        ├── xTest-[species].csv
        ├── yTrain-[species].csv
        └── yTest-[species].csv
```

- **The scripts don't create directories.** Make `collocation/`, `current/`, `split/` and `hyperParams/` under
  `.data/<Location>/` before running. Otherwise the writes fail.
- **`raw.csv`**: QuantAQ minute-level export with newest rows first. Exports can have dozens of columns (OPC bins,
  `*_we`/`*_ae` electrode signals and more). Only `timestamp_local`, `rh`, `temp`, `pm25`, `pm10`, `co`, `no`, `no2`
  and `o3` are read.
- **`gapa-[species].csv`**: Georgia Air Public Access (GAPA) data. Its header must be exactly `Date,[species]`, which
  may mean editing the header of the downloaded file by hand. `Date` looks like `31-May-2025 00:00`
  (`%d-%b-%Y %H:%M`). `o3` and `co` come in ppm, and `preprocess.py` converts them to ppb.
- **Every colocation GAPA file must keep its `Date` column.** It is the only copy of the reference timestamps.
  - `preprocess.py` writes the aligned target to `gapa-preprocessed-[species].csv` and never writes to
    `gapa-[species].csv`. Keep it that way. Writing back over the download with `index=False` drops `Date` and stores
    o3/co already converted to ppb, so the next run fails or double-converts.
  - A `gapa-[species].csv` without a `Date` column was overwritten by an older `preprocess.py`. It can't be
    recovered, so download it again.

## Running the pipeline

The [README](README.md) has the environment setup and the order of steps: preprocess → model bakeoff → train → preprocess
current → visualize. It also has the download window and the GAPA header check. Follow it, running from the location
folder as described above.

The `run-pipeline` skill ([.claude/skills/run-pipeline/SKILL.md](.claude/skills/run-pipeline/SKILL.md)) runs every
step for one location, including choosing the bakeoff winner: `/run-pipeline Pittsburgh no2 pm10`, or all six species
when none are named.

## Model bakeoff

`modelBakeoff.py [species]` chooses which algorithm and hyperparameters `train.py` uses for that species at that
location. It doesn't save a model.

1. It loads `split/{x,y}{Train,Test}-[species].csv`.
2. For each of Decision Tree, Linear, Ridge, Lasso, KNN, Random Forest and XGBoost, it runs `GridSearchCV` and
   `RandomizedSearchCV` (5-fold; the random search tries 33% of the grid). The grids are in `get_parameter_grid()`.
   MLP is commented out. Random Forest and XGBoost are the slowest, at over a minute each per species, and adding values
   to their grids multiplies that quickly. Random Forest's `max_features=1.0` makes it plain bagged trees, so the grid
   covers bagging too.
3. It scores each tuned model on the test split, prints a table of MSE, RMSE, MAE, R² and time, and writes the same
   table to `hyperParams/[species]-scores.csv`. Each row is one model and search, for example `KNN (Grid)`.
4. It writes the best params to `hyperParams/[species].json` as `{"<model>": {"Grid": {...}, "Random": {...}}}`.

The winning row's class and params go into the `species` branch of `train_model()` in that location's `train.py`.
They also go into that location's models table in [docs/featureselection.md](docs/featureselection.md). The winner is
the row with the lowest test RMSE, unless a simpler model (Linear < Ridge/Lasso < KNN < DT < RF < XGB) is within 2% of it. In that
case the simplest of those wins. [pick_model.py](.claude/skills/run-pipeline/scripts/pick_model.py) applies the rule
and prints the line for `train_model()` and the row for the docs. A model added to the bakeoff also needs an entry
in `SIMPLICITY` and `CONSTRUCTORS` there, or the script refuses to pick. The choice is per location and per species, and the
locations already differ. `your_model()` in `modelBakeoff.py` is an unused stub.

## Features differ per sensor and per species

Each low-cost sensor, and each species within it, can keep a different set of training features. Don't assume that
one location's or one species' feature set applies to another.

- **Source of truth.** That location's `preprocess.py` defines the features, grouped by species family:
  - `pm25` and `pm10` each use both PM channels plus coarse-fraction and ratio terms.
  - `no` and `no2` each use both NOx channels.
  - `o3` and `co` use only their own channel.
  - All species add time-of-day and day-of-week terms, rolling means and stds, interactions and squared terms.
  - `rh` and `temp` (and `rh_sq`/`temp_sq` for PM) go through `log1p` and then a per-species `MinMaxScaler`.
- **Per-location lists.** [docs/featureselection.md](docs/featureselection.md) defines every feature. For each
  location, it has a table of which features each species uses and a table of each species' model and
  hyperparameters.
- **Exact list.** The header of `.data/<Location>/split/xTrain-[species].csv` is the exact feature list that a
  location's model for that species was trained on.
- **When features change for a location and species:**
  - Change `preprocess.py` and `preprocessCurrent.py` the same way, producing the same columns in the same order and
    normalizing the same columns. `train.py` fits on DataFrames, so sklearn checks feature names at predict time.
    `visualization.py -f current` feeds every column except `dayhour` to the model, so a mismatch fails there.
  - Rerun preprocess → bakeoff → train → preprocessCurrent for that species. The scaler, split, hyperparameters and
    model all depend on the feature set.
  - Update that location's table in `docs/featureselection.md`.
  - Only copy the change to other locations if it is meant for them too.

## Git

- `models/`, `scalers/` and `figures/` are tracked. Running `preprocess.py`, `train.py` or `visualization.py`
  overwrites those files. Commit them only when you mean to retrain, and leave them out otherwise.
- To add one of the remaining sensors:
  1. Copy an existing location folder.
  2. Replace every `../.data/<Location>` path in the five scripts.
  3. Create the `.data/<Location>/` subfolders and run the pipeline.
