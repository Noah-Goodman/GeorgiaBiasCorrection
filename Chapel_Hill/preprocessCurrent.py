import argparse
import joblib
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(xFeat: pd.DataFrame, species: str):
    """
    Universal preprocess step for current period given a species
    
    Parameters
    ----------
    df : pandas df with shape (n, d)
        Training data
    species: str
        Species to preprocess for

    Returns
    -------
    df: pandas df with shape (n, d)
    """
    if species in ["pm25", "pm10"]:
        xFeat = xFeat[["timestamp_local", "rh", "temp", "pm25", "pm10"]].dropna(subset=["pm25", "pm10"])
    elif species in ["no2", "no"]:
        xFeat = xFeat[["timestamp_local", "rh", "temp", "no2", "no"]].dropna(subset=["no2", "no"])
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
    if species in ["pm25", "pm10"]:
        feat_to_normalize = ["rh", "temp", "rh_sq", "temp_sq"]
    else:
        feat_to_normalize = ["rh", "temp"]
    xFeat[feat_to_normalize] = xFeat[feat_to_normalize].apply(np.log1p)
    scaler = joblib.load(f"scalers/rh-temp-scaler-{species}.pkl")
    xFeat[feat_to_normalize] = scaler.transform(xFeat[feat_to_normalize])

    xFeat = xFeat.dropna()

    return xFeat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("species")
    
    args = parser.parse_args()

    xFeat = pd.read_csv(f"../.data/Chapel-Hill/current/raw.csv")

    xFeat = preprocess_data(xFeat, args.species)
    xFeat.to_csv(f"../.data/Chapel-Hill/current/preprocessed-{args.species}.csv")


if __name__ == "__main__":
    main()