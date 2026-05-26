import argparse
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(xFeat: pd.DataFrame, y: pd.DataFrame):
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
    xFeat = xFeat[["timestamp_local", "rh", "temp", "co"]].dropna(subset=["co"])

    xFeat['timestamp_local'] = pd.to_datetime(xFeat["timestamp_local"])
    xFeat["dayhour"] = xFeat["timestamp_local"].dt.strftime('%Y-%m-%d %H')
    xFeat["dayhour"].sort_index()

    
    # average data by dayhour
    xFeat = xFeat.groupby("dayhour").agg(
        co=("co", lambda x: x.mean(skipna=True)),
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

    # Rolling features for PM10
    xFeat['co_roll_3h'] = xFeat['co'].rolling(window=3, min_periods=1).mean()
    xFeat['co_roll_6h'] = xFeat['co'].rolling(window=6, min_periods=1).mean()
    xFeat['co_std_3h'] = xFeat['co'].rolling(window=3, min_periods=1).std().fillna(0)

    # Interaction terms - these capture how RH/temp affect bias differently at different PM levels
    xFeat['co_x_rh'] = xFeat['co'] * xFeat['rh']
    xFeat['co_x_temp'] = xFeat['co'] * xFeat['temp']
    xFeat['rh_x_temp'] = xFeat['rh'] * xFeat['temp']
    
    # Polynomial terms - capture non-linear effects
    xFeat['co_sq'] = xFeat['co'] ** 2
    # xFeat['rh_sq'] = xFeat['rh'] ** 2
    # xFeat['temp_sq'] = xFeat['temp'] ** 2

    # PM10/PM2.5 ratio
    # xFeat['pm25_pm10_ratio'] = xFeat['pm25'] / (xFeat['pm10'] + 0.001)
    # # Calculate coarse fraction early
    # xFeat['coarse_fraction'] = (xFeat['pm10'] - xFeat['pm25']).clip(lower=0)
    # xFeat['coarse_x_rh'] = xFeat['coarse_fraction'] * xFeat['rh']
    # xFeat['coarse_x_temp'] = xFeat['coarse_fraction'] * xFeat['temp']
    # xFeat['coarse_sq'] = xFeat['coarse_fraction'] ** 2
    # # Coarse ratio (tells you if it's dusty or combustion-dominated)
    # xFeat['coarse_ratio'] = xFeat['coarse_fraction'] / (xFeat['pm10'] + 0.001)
    

    # normalize rh and temp
    feat_to_normalize = ["rh", "temp"]
    xFeat[feat_to_normalize] = xFeat[feat_to_normalize].apply(np.log1p)
    scaler = MinMaxScaler()
    xFeat[feat_to_normalize] = scaler.fit_transform(xFeat[feat_to_normalize])

    # y
    y.dropna(subset=["co"], inplace=True)
    y['Date'] = pd.to_datetime(y["Date"], format='%d-%b-%Y %H:%M')
    y['dayhour'] = y["Date"].dt.strftime('%Y-%m-%d %H')
    y.drop("Date", axis=1, inplace=True)
    y.set_index("dayhour",inplace=True)


    # only keep rows that have dayhour in gapa DF
    common_dayhours = set(xFeat.index).intersection(set(y.index))
    xFeat = xFeat[xFeat.index.isin(common_dayhours)]
    y = y[y.index.isin(common_dayhours)]

    return xFeat, y

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
    parser.add_argument("xFeat")
    parser.add_argument("y")
    
    args = parser.parse_args()

    xFeat = pd.read_csv(args.xFeat)
    y = pd.read_csv(args.y)

    xFeat, y = preprocess_data(xFeat, y)
    xFeat.to_csv("Preprocessed-00589.csv")
    y.to_csv("Preprocessed-gapa.csv")

    xTrain, xTest, yTrain, yTest = holdout(xFeat, y)
    xTrain.to_csv("xTrain.csv", index=False)
    xTest.to_csv("xTest.csv", index=False)
    yTrain.to_csv("yTrain.csv", index=False)
    yTest.to_csv("yTest.csv", index=False)



if __name__ == "__main__":
    main()