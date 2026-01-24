import argparse
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import joblib

def preprocess_data(xFeat: pd.DataFrame, y: pd.DataFrame, species: str):
    """
    Universal preprocess step for collocation period given a species
    
    Parameters
    ----------
    xFeat : pandas df with shape (n, d)
        Training data
    y : pandas df with shape (n, d)
        Target GAPA data
    species: str
        Species to preprocess for

    Returns
    -------
    df: pandas df with shape (n, d)
    """

    # Universal preprocessing
    if species in ["pm25", "pm10"]:
        xFeat = xFeat[["timestamp_local", "rh", "temp", "pm25", "pm10"]].dropna(subset=[species])
    elif species in ["no2", "no"]:
        xFeat = xFeat[["timestamp_local", "rh", "temp", "no2", "no"]].dropna(subset=[species])
    else:
        xFeat = xFeat[["timestamp_local", "rh", "temp", species]].dropna(subset=[species])


    xFeat['timestamp_local'] = pd.to_datetime(xFeat["timestamp_local"])
    xFeat["dayhour"] = xFeat["timestamp_local"].dt.strftime('%Y-%m-%d %H')
    xFeat["dayhour"].sort_index()

    
    # average data by dayhour
    if species in ["pm25", "pm10"]:
        xFeat = xFeat.groupby("dayhour").agg(
            pm25=("pm25", lambda x: x.mean(skipna=True)),
            pm10=("pm10", lambda x: x.mean(skipna=True)),
            rh=("rh", lambda x: x.mean(skipna=True)),
            temp=("temp", lambda x: x.mean(skipna=True))
        )
    elif species in ["no2", "no"]:
        xFeat = xFeat.groupby("dayhour").agg(
            no2=("no2", lambda x: x.mean(skipna=True)),
            no=("no", lambda x: x.mean(skipna=True)),
            rh=("rh", lambda x: x.mean(skipna=True)),
            temp=("temp", lambda x: x.mean(skipna=True))
        )
    else:
        xFeat = xFeat.groupby("dayhour").agg(
            **{species: (species, lambda x: x.mean(skipna=True))},
            rh=("rh", lambda x: x.mean(skipna=True)),
            temp=("temp", lambda x: x.mean(skipna=True))
        )

    # Temporal features
    xFeat["day_of_week"] = pd.to_datetime(xFeat.index).dayofweek
    xFeat["hour"] = pd.to_datetime(xFeat.index).hour
    xFeat['hour_sin'] = np.sin(2 * np.pi * xFeat['hour'] / 24)
    xFeat['hour_cos'] = np.cos(2 * np.pi * xFeat['hour'] / 24)
    xFeat['day_sin'] = np.sin(2 * np.pi * xFeat['day_of_week'] / 7)
    xFeat['day_cos'] = np.cos(2 * np.pi * xFeat['day_of_week'] / 7)
    xFeat.drop(["hour"], axis=1, inplace=True)

    # Rolling features
    xFeat[f'{species}_roll_3h'] = xFeat[species].rolling(window=3, min_periods=1).mean()
    xFeat[f'{species}_roll_6h'] = xFeat[species].rolling(window=6, min_periods=1).mean()
    xFeat[f'{species}_std_3h'] = xFeat[species].rolling(window=3, min_periods=1).std().fillna(0)

    # Interaction terms - these capture how RH/temp affect bias differently at different PM levels
    xFeat[f'{species}_x_rh'] = xFeat[species] * xFeat['rh']
    xFeat[f'{species}_x_temp'] = xFeat[species] * xFeat['temp']
    xFeat['rh_x_temp'] = xFeat['rh'] * xFeat['temp']
    
    # Polynomial terms - capture non-linear effects
    xFeat[f'{species}_sq'] = xFeat[species] ** 2
    if species in ["pm25", "pm10"]:
        xFeat['rh_sq'] = xFeat['rh'] ** 2
        xFeat['temp_sq'] = xFeat['temp'] ** 2

    # PM10/PM2.5 ratio
    if species in ["pm25", "pm10"]:
        xFeat['pm25_pm10_ratio'] = xFeat['pm25'] / (xFeat['pm10'] + 0.001)
        # Calculate coarse fraction early
        xFeat['coarse_fraction'] = (xFeat['pm10'] - xFeat['pm25']).clip(lower=0)
        xFeat['coarse_x_rh'] = xFeat['coarse_fraction'] * xFeat['rh']
        xFeat['coarse_x_temp'] = xFeat['coarse_fraction'] * xFeat['temp']
        xFeat['coarse_sq'] = xFeat['coarse_fraction'] ** 2
        # Coarse ratio (tells you if it's dusty or combustion-dominated)
        xFeat['coarse_ratio'] = xFeat['coarse_fraction'] / (xFeat['pm10'] + 0.001)
    

    # normalize rh and temp
    feat_to_normalize = ["rh", "temp", "rh_sq", "temp_sq"]
    xFeat[feat_to_normalize] = xFeat[feat_to_normalize].apply(np.log1p)
    scaler = MinMaxScaler()
    xFeat[feat_to_normalize] = scaler.fit_transform(xFeat[feat_to_normalize])

    # y
    y.dropna(subset=[species], inplace=True)
    y['Date'] = pd.to_datetime(y["Date"])
    y['dayhour'] = y["Date"].dt.strftime('%Y-%m-%d %H')
    y.drop("Date", axis=1, inplace=True)
    y.set_index("dayhour",inplace=True)


    # only keep rows that have dayhour in gapa DF
    common_dayhours = set(xFeat.index).intersection(set(y.index))
    xFeat = xFeat[xFeat.index.isin(common_dayhours)]
    y = y[y.index.isin(common_dayhours)]

    return xFeat, y, scaler

def holdout(xFeat, y, testSize=0.2):
    rng = np.random.default_rng(seed=10)
    N = len(xFeat)
    permuted_indices = rng.permutation(N)
    xFeat = xFeat.iloc[permuted_indices]
    y = y.iloc[permuted_indices]

    split = int(N*(1-testSize))
    xTrain = xFeat.iloc[:split]
    xTest = xFeat.iloc[split:]
    yTrain = y.iloc[:split]
    yTest = y.iloc[split:]

    return xTrain, xTest, yTrain, yTest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species")
    
    args = parser.parse_args()

    xFeat = pd.read_csv(f"../.data/Chapel-Hill/collocation/raw.csv")
    y = pd.read_csv(f"../.data/Chapel-Hill/collocation/gapa-{args.species}.csv")

    xFeat, y, scaler = preprocess_data(xFeat, y, args.species)
    # Save dayhours used for visualization
    xFeat.index.to_series().to_csv(f"../.data/Chapel-Hill/dayhour/{args.species}.csv", index=False)

    joblib.dump(scaler, f"scalers/rh-temp-scaler-{args.species}.pkl")

    xTrain, xTest, yTrain, yTest = holdout(xFeat, y)
    xTrain.to_csv(f"../.data/Chapel-Hill/split/xTrain-{args.species}.csv", index=False)
    xTest.to_csv(f"../.data/Chapel-Hill/split/xTest-{args.species}.csv", index=False)
    yTrain.to_csv(f"../.data/Chapel-Hill/split/yTrain-{args.species}.csv", index=False)
    yTest.to_csv(f"../.data/Chapel-Hill/split/yTest-{args.species}.csv", index=False)


if __name__ == "__main__":
    main()