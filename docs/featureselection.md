# Feature selection

This doc lists the features each location's model is trained on, for each of the six species, along with the model
class and hyperparameters used. Each sensor, and each species within it, can keep its own features and model, so
each location has its own tables.

`preprocess_data()` builds the features. It lives in each location's `preprocess.py` (colocation data) and
`preprocessCurrent.py` (current data). The two must build the same columns in the same order. Models are fit on
DataFrames, so sklearn rejects current data whose column names or order differ from training.

The features of a trained model are the columns of `.data/<Location>/split/xTrain-[species].csv`. They are also the
`feature_names_in_` of `<Location>/models/[species]-model.pkl`. Neither the target (the GAPA value) nor the `dayhour`
index is a feature.

## How features are built

1. **Select channels.** Keep `timestamp_local`, `rh`, `temp` and the species' sensor channels. Drop rows missing any
   of those channels.
2. **Average to hourly.** Group by `dayhour` (`YYYY-MM-DD HH`) and take the mean of each channel.
3. **Derive features.** Add time, rolling, interaction and squared terms. PM species also get ratio and
   coarse-fraction terms.
4. **Scale humidity and temperature.** Apply `log1p` and then a `MinMaxScaler` to `rh` and `temp`, plus `rh_sq` and
   `temp_sq` for PM. `preprocess.py` fits the scaler on the colocation data and saves it to
   `scalers/rh-temp-scaler-[species].pkl`. `preprocessCurrent.py` loads it and only transforms.

Some features are added only for certain species. These are the conditions in `preprocess_data()`:

| Condition                      | What it adds                                                                                      |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| species is `pm25` or `pm10`    | Both PM channels, `rh_sq` and `temp_sq` (scaled with `rh` and `temp`), `pm25_pm10_ratio` and the `coarse_*` terms |
| species is `no` or `no2`       | Both NOx channels                                                                                 |
| any other species (`o3`, `co`) | Only that species' own channel                                                                    |

## Feature definitions

`{s}` is the species being trained. For example, the `no2` model has `no2_roll_3h`.

| Feature                        | Used for   | Definition                                                    | Scaled         | Why it's included |
| ------------------------------ | ---------- | ------------------------------------------------------------- | -------------- | ----------------- |
| `pm25`, `pm10`                 | pm25, pm10 | Hourly mean of the sensor channel                             | No             | The raw reading being corrected. The other PM channel adds particle-size information, and optical sensor error depends on particle size. |
| `no2`, `no`                    | no, no2    | Hourly mean of the sensor channel                             | No             | The raw reading being corrected. NO converts to NO₂ in the air, so the other channel says something about the one being corrected. |
| `o3`                           | o3         | Hourly mean of the sensor channel                             | No             | The raw reading being corrected. |
| `co`                           | co         | Hourly mean of the sensor channel                             | No             | The raw reading being corrected. |
| `rh`                           | All        | Hourly mean relative humidity                                 | log1p + MinMax | Humidity is a main driver of low-cost sensor bias. Particles take up water and read high on optical sensors, and electrochemical cells respond to humidity changes. |
| `temp`                         | All        | Hourly mean temperature                                       | log1p + MinMax | Electrochemical sensors' baseline and sensitivity drift with temperature. |
| `day_of_week`                  | All        | 0 = Monday … 6 = Sunday                                       | No             | Emissions follow a weekly cycle, such as weekday traffic versus weekends. |
| `hour_sin`, `hour_cos`         | All        | sin and cos of 2π · hour / 24                                 | No             | Pollution follows a daily cycle (rush hour, boundary-layer height, ozone photochemistry). Encoding hour as a circle keeps 23:00 next to 00:00. |
| `day_sin`, `day_cos`           | All        | sin and cos of 2π · day_of_week / 7                           | No             | The weekly cycle on a circle, so Sunday sits next to Monday. |
| `{s}_roll_3h`, `{s}_roll_6h`   | All        | Rolling mean of `{s}` over the last 3 or 6 rows               | No             | Smooths sensor noise and gives the model recent history. |
| `{s}_std_3h`                   | All        | Rolling std of `{s}` over the last 3 rows, 0 on the first row | No             | Flags spikes and unstable stretches, where bias tends to be larger. |
| `{s}_x_rh`, `{s}_x_temp`       | All        | `{s}` × `rh` and `{s}` × `temp`                               | No             | Humidity and temperature affect bias differently at different concentrations. |
| `rh_x_temp`                    | All        | `rh` × `temp`                                                 | No             | Lets the effect of humidity depend on temperature, and the other way round. |
| `{s}_sq`                       | All        | `{s}`²                                                        | No             | Sensor response isn't linear in concentration. |
| `rh_sq`, `temp_sq`             | pm25, pm10 | `rh`² and `temp`²                                             | log1p + MinMax | Water uptake by particles rises sharply at high humidity, so the humidity effect on PM isn't linear. |
| `pm25_pm10_ratio`              | pm25, pm10 | `pm25` / (`pm10` + 0.001)                                     | No             | The share of fine particles, which shifts with the source and changes optical sensor bias. The 0.001 avoids dividing by zero. |
| `coarse_fraction`              | pm25, pm10 | max(`pm10` − `pm25`, 0)                                       | No             | Coarse particles such as dust and pollen are measured less accurately by optical sensors. |
| `coarse_x_rh`, `coarse_x_temp` | pm25, pm10 | `coarse_fraction` × `rh` and `coarse_fraction` × `temp`       | No             | Humidity and temperature affect coarse-particle bias differently at different coarse loads. |
| `coarse_sq`                    | pm25, pm10 | `coarse_fraction`²                                            | No             | The coarse-particle effect isn't linear. |
| `coarse_ratio`                 | pm25, pm10 | `coarse_fraction` / (`pm10` + 0.001)                          | No             | Tells dusty conditions from combustion-dominated ones. |

- **Interaction and squared terms use raw `rh` and `temp`.** They are computed before scaling.
- **Rolling windows count rows, not clock hours.** A row is an hour that has data, so a window can span a gap in the
  record. The windows are computed over all of the sensor's hours, before rows are matched to GAPA.
- **Each PM model sees both PM channels, and each NOx model sees both NOx channels.** For example, `pm10` is an input
  to the `pm25` model.

## Per-location features and models

In each features table, a ✓ means that species' model uses the feature. Reading a species' ✓ rows from top to bottom
gives its exact column order.

Each models table comes from `train_model()` in that location's `train.py`. The model was picked with
`modelBakeoff.py` and saved to `models/[species]-model.pkl`. Hyperparameters not listed use the sklearn (or, for `XGBRegressor`, the xgboost) defaults.
`p` only applies when `metric="minkowski"`, so it has no effect on these KNN models, which all use
`metric="manhattan"`.

### Chapel Hill (`Chapel_Hill/`, MOD-00592)

#### Features

| Feature           | pm25 | pm10 | no | no2 | o3 | co |
| ----------------- | :--: | :--: | :-: | :-: | :-: | :-: |
| `pm25`            | ✓ | ✓ |   |   |   |   |
| `pm10`            | ✓ | ✓ |   |   |   |   |
| `no2`             |   |   | ✓ | ✓ |   |   |
| `no`              |   |   | ✓ | ✓ |   |   |
| `o3`              |   |   |   |   | ✓ |   |
| `co`              |   |   |   |   |   | ✓ |
| `rh`              | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `temp`            | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_of_week`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour_sin`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour_cos`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_sin`         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_cos`         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_roll_3h`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_roll_6h`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_std_3h`      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_x_rh`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_x_temp`      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rh_x_temp`       | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_sq`          | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rh_sq`           | ✓ | ✓ |   |   |   |   |
| `temp_sq`         | ✓ | ✓ |   |   |   |   |
| `pm25_pm10_ratio` | ✓ | ✓ |   |   |   |   |
| `coarse_fraction` | ✓ | ✓ |   |   |   |   |
| `coarse_x_rh`     | ✓ | ✓ |   |   |   |   |
| `coarse_x_temp`   | ✓ | ✓ |   |   |   |   |
| `coarse_sq`       | ✓ | ✓ |   |   |   |   |
| `coarse_ratio`    | ✓ | ✓ |   |   |   |   |
| **Total**         | 24 | 24 | 16 | 16 | 15 | 15 |

#### Models

| Species | Model                   | Hyperparameters |
| ------- | ----------------------- | --------------- |
| pm25    | `DecisionTreeRegressor` | `criterion="absolute_error", min_samples_leaf=4, min_samples_split=2, max_depth=20, splitter="best", random_state=42` |
| pm10    | `DecisionTreeRegressor` | `criterion="absolute_error", min_samples_leaf=4, min_samples_split=2, max_depth=15, splitter="random", random_state=42` |
| no      | `KNeighborsRegressor`   | `weights="distance", n_neighbors=7, p=2, metric="manhattan"` |
| no2     | `Lasso`                 | `alpha=0.001, max_iter=5000, fit_intercept=False` |
| o3      | `Lasso`                 | `alpha=0.001, max_iter=5000, fit_intercept=False` |
| co      | `DecisionTreeRegressor` | `criterion="squared_error", min_samples_leaf=4, min_samples_split=10, max_depth=15, splitter="random", random_state=42` |

### Lithonia (`Lithonia/`, MOD-00595)

#### Features

| Feature           | pm25 | pm10 | no | no2 | o3 | co |
| ----------------- | :--: | :--: | :-: | :-: | :-: | :-: |
| `pm25`            | ✓ | ✓ |   |   |   |   |
| `pm10`            | ✓ | ✓ |   |   |   |   |
| `no2`             |   |   | ✓ | ✓ |   |   |
| `no`              |   |   | ✓ | ✓ |   |   |
| `o3`              |   |   |   |   | ✓ |   |
| `co`              |   |   |   |   |   | ✓ |
| `rh`              | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `temp`            | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_of_week`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour_sin`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour_cos`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_sin`         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_cos`         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_roll_3h`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_roll_6h`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_std_3h`      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_x_rh`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_x_temp`      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rh_x_temp`       | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_sq`          | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rh_sq`           | ✓ | ✓ |   |   |   |   |
| `temp_sq`         | ✓ | ✓ |   |   |   |   |
| `pm25_pm10_ratio` | ✓ | ✓ |   |   |   |   |
| `coarse_fraction` | ✓ | ✓ |   |   |   |   |
| `coarse_x_rh`     | ✓ | ✓ |   |   |   |   |
| `coarse_x_temp`   | ✓ | ✓ |   |   |   |   |
| `coarse_sq`       | ✓ | ✓ |   |   |   |   |
| `coarse_ratio`    | ✓ | ✓ |   |   |   |   |
| **Total**         | 24 | 24 | 16 | 16 | 15 | 15 |

#### Models

| Species | Model                 | Hyperparameters |
| ------- | --------------------- | --------------- |
| pm25    | `LinearRegression`    | `fit_intercept=True, copy_X=True` |
| pm10    | `LinearRegression`    | `fit_intercept=False` |
| no      | `KNeighborsRegressor` | `weights="distance", n_neighbors=11, metric="manhattan"` |
| no2     | `Ridge`               | `alpha=1.0, fit_intercept=True, random_state=42` |
| o3      | `Ridge`               | `alpha=1.0, fit_intercept=True, random_state=42` |
| co      | `KNeighborsRegressor` | `weights="uniform", n_neighbors=15, p=2, metric="manhattan"` |

### Pittsburgh (`Pittsburgh/`, MOD-00591)

#### Features

| Feature           | pm25 | pm10 | no | no2 | o3 | co |
| ----------------- | :--: | :--: | :-: | :-: | :-: | :-: |
| `pm25`            | ✓ | ✓ |   |   |   |   |
| `pm10`            | ✓ | ✓ |   |   |   |   |
| `no2`             |   |   | ✓ | ✓ |   |   |
| `no`              |   |   | ✓ | ✓ |   |   |
| `o3`              |   |   |   |   | ✓ |   |
| `co`              |   |   |   |   |   | ✓ |
| `rh`              | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `temp`            | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_of_week`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour_sin`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hour_cos`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_sin`         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `day_cos`         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_roll_3h`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_roll_6h`     | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_std_3h`      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_x_rh`        | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_x_temp`      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rh_x_temp`       | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `{s}_sq`          | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `rh_sq`           | ✓ | ✓ |   |   |   |   |
| `temp_sq`         | ✓ | ✓ |   |   |   |   |
| `pm25_pm10_ratio` | ✓ | ✓ |   |   |   |   |
| `coarse_fraction` | ✓ | ✓ |   |   |   |   |
| `coarse_x_rh`     | ✓ | ✓ |   |   |   |   |
| `coarse_x_temp`   | ✓ | ✓ |   |   |   |   |
| `coarse_sq`       | ✓ | ✓ |   |   |   |   |
| `coarse_ratio`    | ✓ | ✓ |   |   |   |   |
| **Total**         | 24 | 24 | 16 | 16 | 15 | 15 |

#### Models

| Species | Model                   | Hyperparameters |
| ------- | ----------------------- | --------------- |
| pm25    | `DecisionTreeRegressor` | `criterion="absolute_error", min_samples_leaf=4, min_samples_split=2, max_depth=20, splitter="best", random_state=42` |
| pm10    | `RandomForestRegressor` | `max_depth=None, max_features=0.33, min_samples_leaf=1, n_estimators=300, random_state=42` |
| no      | `XGBRegressor`          | `learning_rate=0.03, max_depth=6, min_child_weight=1, n_estimators=100, random_state=42, subsample=0.7` |
| no2     | `RandomForestRegressor` | `max_depth=None, max_features=0.5, min_samples_leaf=2, n_estimators=300, random_state=42` |
| o3      | `Ridge`                 | `alpha=1.0, fit_intercept=True, random_state=42` |
| co      | `KNeighborsRegressor`   | `weights="uniform", n_neighbors=15, p=2, metric="manhattan"` |

## Keeping this doc current

When you change the features or the model for a location and species, update that location's tables in the same
commit. To check the tables against a trained model, run this from the location folder. It prints the model with its
non-default hyperparameters, followed by its features:

```bash
python -c "import joblib; m = joblib.load('models/no2-model.pkl'); print(m); print(list(m.feature_names_in_))"
```
