---
name: run-pipeline
description: Run the bias-correction pipeline end to end for one sensor location (Chapel_Hill, Lithonia, Pittsburgh or any other location folder) - preprocess, model bakeoff, choose the winning model and hyperparameters and write them into train.py, train, preprocess current data, and plot both the test split and the corrected current data. Use whenever the user wants to retrain a location's models, rerun the bakeoff, pick models from bakeoff results, process newly downloaded QuantAQ or GAPA data, or regenerate the correction figures, for one species or all six, even if they don't say "pipeline".
argument-hint: <Location> [species ...]
---

# Run the bias-correction pipeline

This takes one location and a list of species through every step in the README. It also does the step that used to
be manual: picking the bakeoff winner and copying its class and hyperparameters into `train.py`.

Arguments: `$ARGUMENTS`. The first word is the location and the rest are species. When no species are given, run all
six: `pm25 pm10 no no2 o3 co`. If no location was given, ask for one rather than guessing. Match the location loosely
("chapel hill" → `Chapel_Hill`) against the top-level folders that contain a `preprocess.py`.

Do steps 1–3 for every species first, so the slow bakeoffs run side by side. Then do steps 4–7 one species at a time.
If the user asked for only part of the pipeline (say, "retrain no2 from the bakeoff I already ran"), run just those
steps in this order, after checking that the files each step reads exist.

## Ground rules

- **Run each script from inside its location folder, with the repo's venv.** The scripts read `../.data/<Location>/...`
  and write `models/`, `scalers/` and `figures/` relative to the folder they run in. Shell state doesn't carry over
  between commands, so put the `cd` in every command: `cd <repo>/<Location> && ../.venv/bin/python preprocess.py no2`.
  If `.venv/` is missing, set it up the way the README describes.
- **Get the data folder from the scripts, not from the folder name.** Code folders use underscores and data folders use
  hyphens (`Chapel_Hill` → `.data/Chapel-Hill`). `grep -m1 -o '\.\./\.data/[^/]*' <Location>/preprocess.py` prints
  the exact path. Below, `<Data>` stands for that data folder's name.
- **Run the pipeline and record the chosen model, nothing else.** The only edits are the model line in `train.py` and
  its row in `docs/featureselection.md`. Leave features, parameter grids and plotting as they are.
  Feature changes have their own procedure in AGENTS.md: a change made here would silently make the other species'
  scalers and models out of date.
- **Never write to the downloaded files** (`raw.csv`, `gapa-[species].csv`). Each GAPA file is the only copy of its
  reference timestamps.
- **Don't commit.** Preprocessing, training and plotting overwrite tracked files in `scalers/`, `models/` and
  `figures/`. The user reviews the changes and decides whether to commit them.
- **If one species fails, keep going with the others.** Skip that species and give the reason in the report.

## 1. Check inputs

Create the folders the scripts write to, which they don't create themselves:

```bash
mkdir -p <repo>/.data/<Data>/{collocation,current,split,hyperParams}
```

Then check, for each species:

- `.data/<Data>/collocation/raw.csv` exists. All species share it.
- `.data/<Data>/collocation/gapa-<species>.csv` exists and its first line is exactly `Date,<species>`, ignoring a
  trailing `\r` or a leading BOM. Skip the species if either check fails, and tell the user which file failed and what
  its header says:
  - If the header has a `Date` column but a different species column name, the user fixes the header by hand (README,
    "Data cleaning").
  - If there is no `Date` column, an older `preprocess.py` overwrote the file, and it has to be downloaded again.
- `.data/<Data>/current/raw.csv` exists. Without it the species can still be trained. Skip step 6 and
  `visualization.py -f current`, and say so in the report.

## 2. Preprocess

```bash
cd <repo>/<Location> && ../.venv/bin/python preprocess.py <species>
```

This takes seconds. It writes the hourly features, the GAPA target aligned to them, the 80/20 split in `split/` and
`scalers/rh-temp-scaler-<species>.pkl`.

## 3. Bakeoff

```bash
cd <repo>/<Location> && ../.venv/bin/python modelBakeoff.py <species>
```

This is the slow step, and Random Forest and XGBoost take most of it. For Pittsburgh's no, no2 and pm10 run side by
side, RF's Grid search took 72–121 s and its Random search 24–39 s, and XGB's took 100–128 s and 31–38 s (the scores
csv's Time column). The other five models took 76–123 s together, most of it Ridge's Grid search. The species don't
depend on each other, so run all of their bakeoffs at once. Each bakeoff runs its fits one at a time, and RF and XGB on
a single thread, so they barely compete for cores: on an 8-core Mac, those three finished in 7.5 minutes. Start them
in one background command that saves each species' output to its own log in your scratchpad, so you can check them
separately once they finish:

```bash
cd <repo>/<Location> && for s in <species ...>; do ../.venv/bin/python modelBakeoff.py $s > <scratchpad>/bakeoff-$s.log 2>&1 & done; wait
```

Each bakeoff overwrites the location's previous `hyperParams/<species>.json` and `<species>-scores.csv`. Those files
aren't in git, so if the user wants the old scores to compare against, copy them somewhere first. To try a change to
`modelBakeoff.py` without touching the location's results, copy the script into `<scratchpad>/<Location>/` and the
split into `<scratchpad>/.data/<Data>/split/`, create `<scratchpad>/.data/<Data>/hyperParams/`, and run it from
`<scratchpad>/<Location>`. Its `../.data/...` paths then resolve inside the scratchpad.

Each bakeoff writes two files:

- `hyperParams/<species>.json`: the best params for each model and search.
- `hyperParams/<species>-scores.csv`: test MSE, RMSE, MAE, R² and time for each of the 14 rows (7 models × Grid and
  Random search, for example `KNN (Grid)`).

Expect hundreds of `ConvergenceWarning`s from Ridge and Lasso, about 550–700 per species. They don't mean the run
failed. A missing scores file after the run does: read that bakeoff's log to find out why.

A `FitFailedWarning` means a value in `get_parameter_grid()` is one the installed sklearn rejects. Those fits score
NaN, so they can't win, but mention it in the report so the grid can be fixed.

## 4. Pick the model

```bash
cd <repo>/<Location> && ../.venv/bin/python ../.claude/skills/run-pipeline/scripts/pick_model.py ../.data/<Data>/hyperParams <species>
```

The script applies the rule the lab chose: **the lowest test RMSE wins, unless a simpler model (Linear < Ridge/Lasso <
KNN < DT < RF < XGB) is within 2% of it. In that case the simplest of those wins.** The reasons:

- Every candidate is scored on the same `yTest`, so R², MSE and RMSE rank them identically, and RMSE stands in for all
  three. Squared error weights the pollution peaks most, and those are the hours that matter most.
- The test splits are about 200 hours, so a gap of 1–2% is mostly noise. Linear, Ridge and Lasso often land that close
  together, and so do the Grid and Random rows of one model.
- Linear models extrapolate past the colocation range. KNN, DT, RF and XGB can't predict above the highest value they saw in
  training, and current data comes from a different place and season.

Use the script's pick even if another row looks better to you. That way every run and every contributor gets the same
answer from the same scores. If something looks wrong, such as an R² below 0 or every model scoring poorly, raise it
in the report instead.

Also compare each param the script prints with its values in `get_parameter_grid()`. A pick at the lowest or highest
value of a grid list (say `learning_rate=0.03` from `[0.03, 0.1, 0.3]`) means the best value may lie outside the grid.
Pittsburgh's XGB picks landed on the lowest `learning_rate` for all three species and on the most trees or the deepest
trees for two each, and both of its RF picks used the most trees (`n_estimators=300`). Note those params in the report, but leave the grid alone: changing it is a separate change, and
every species at every location would have to be rerun.

The script prints the line for `train.py` and the matching row for the docs. For example:

```
Line for this species' branch of train_model():
    model = KNeighborsRegressor(metric='manhattan', n_neighbors=7, p=2, weights='distance')

Row for the location's models table in docs/featureselection.md:
| no | `KNeighborsRegressor` | `metric="manhattan", n_neighbors=7, p=2, weights="distance"` |
```

Both copy every tuned param from that row's block in the JSON. For DT, Ridge, Lasso, RF and XGB they also add
`random_state=42`, the same as the bakeoff used. The bakeoff also runs XGB with `n_jobs=1`, which only affects speed,
so the printed line leaves it out.

## 5. Record it and train

In `<Location>/train.py`, `train_model()` has one `if`/`elif species == "<species>":` branch per species. Replace the
uncommented `model = ...` line in that branch with the line the script printed. The same model line can appear in more
than one branch (Chapel_Hill's no2 and o3 branches are identical), so include the `elif species == "<species>":` line
in the text you replace to make sure you edit the right branch. Leave `model.fit(...)`, the other branches and any
commented-out lines alone. Don't add a comment about the model being replaced: the git diff records that. The five
original sklearn model classes are already imported. If one isn't, add it to the existing import line.
`RandomForestRegressor` and `XGBRegressor` aren't imported until a species picks them: then add
`from sklearn.ensemble import RandomForestRegressor` below the other sklearn imports, or
`from xgboost import XGBRegressor` below all of them.

In `docs/featureselection.md`, find the location's section (for example `### Chapel Hill`) and replace that species' row in its `#### Models` table with the row the script printed. The paragraph above the
location sections says `p` has no effect because every KNN model uses `metric="manhattan"`. If the new pick is a KNN
with a different metric, correct that paragraph too, so it stays true. AGENTS.md asks for the table to be updated
alongside `train.py`, so that the docs always describe the models that are committed.

```bash
cd <repo>/<Location> && ../.venv/bin/python train.py <species>
```

This saves `models/<species>-model.pkl` and prints the test MSE and R². The data, params and seeds are the same as for
the bakeoff row, so these numbers should match the ones `pick_model.py` printed to about 4 significant figures. If they
don't, the wrong branch was edited or a param was copied wrong. Fix it and retrain before going on.

## 6. Preprocess current data

```bash
cd <repo>/<Location> && ../.venv/bin/python preprocessCurrent.py <species>
```

This loads the scaler that step 2 saved, so it always runs after preprocessing, never before.

## 7. Visualize

```bash
cd <repo>/<Location> && ../.venv/bin/python visualization.py <species> -f train
cd <repo>/<Location> && ../.venv/bin/python visualization.py <species> -f current
```

The figures go in `<Location>/figures/`, named after the plot title. Titles differ between locations, so use
`ls -t figures | head` to find the new files. Open each one (Read works on PNGs) and check two things:

- On the train plot, the corrected line should follow FRM/FEM more closely than the raw line does.
- On the current plot, look for corrected values below zero, or a corrected line that goes flat at the top. A flat top
  means KNN, DT, RF or XGB is capping at the colocation maximum.

Report anything like that, but don't change the model because of it.

## Report

End with one table covering every species that ran:

| Species | Model (search) | Hyperparameters | Bakeoff RMSE / R² / MAE | Lowest-RMSE row, if different | train.py matches | Figures |
| ------- | -------------- | --------------- | ----------------------- | ----------------------------- | ---------------- | ------- |

After the table:

- Each skipped species, and the reason.
- Warnings: an R² below 0, a train.py result that didn't match the bakeoff, a picked param at the edge of its grid,
  anything odd in the figures.
- The tracked files that changed, from `git status --short <Location> docs` (`train.py`, `models/`, `scalers/`,
  `figures/`, `docs/featureselection.md`), and a reminder that nothing was committed.
