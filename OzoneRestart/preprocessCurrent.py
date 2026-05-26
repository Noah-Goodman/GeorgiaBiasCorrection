import argparse
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(xFeat: pd.DataFrame):
    """
    Preprocess the features
    
    Parameters
    ----------
    df : pandas df with shape (n, d)
        Training data

    Returns
    -------
    df: pandas df with shape (n, d)
    """
    xFeat = xFeat[["timestamp_local", "rh", "temp", "o3"]].dropna(subset=["o3"])

    xFeat['timestamp_local'] = pd.to_datetime(xFeat["timestamp_local"])
    xFeat["dayhour"] = xFeat["timestamp_local"].dt.strftime('%Y-%m-%d %H')
    xFeat["dayhour"].sort_index()

    
    # average data by dayhour
    xFeat = xFeat.groupby("dayhour").agg(
        o3=("o3", lambda x: x.mean(skipna=True)),
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

    # Rolling features for o3
    xFeat['o3_roll_3h'] = xFeat['o3'].rolling(window=3, min_periods=1).mean()
    xFeat['o3_roll_6h'] = xFeat['o3'].rolling(window=6, min_periods=1).mean()
    xFeat['o3_std_3h'] = xFeat['o3'].rolling(window=3, min_periods=1).std().fillna(0)

    # Interaction terms - these capture how RH/temp affect bias differently at different PM levels
    xFeat['o3_x_rh'] = xFeat['o3'] * xFeat['rh']
    xFeat['o3_x_temp'] = xFeat['o3'] * xFeat['temp']
    xFeat['rh_x_temp'] = xFeat['rh'] * xFeat['temp']
    
    # Polynomial terms - capture non-linear effects
    xFeat['o3_sq'] = xFeat['o3'] ** 2
    

    # normalize rh and temp
    feat_to_normalize = ["rh", "temp"]
    xFeat[feat_to_normalize] = xFeat[feat_to_normalize].apply(np.log1p)
    scaler = MinMaxScaler()
    xFeat[feat_to_normalize] = scaler.fit_transform(xFeat[feat_to_normalize])

    return xFeat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xFeat")
    parser.add_argument("outFile")
    
    args = parser.parse_args()

    xFeat = pd.read_csv(args.xFeat)
    outFile = args.outFile

    xFeat = preprocess_data(xFeat)
    xFeat.to_csv(outFile)


if __name__ == "__main__":
    main()