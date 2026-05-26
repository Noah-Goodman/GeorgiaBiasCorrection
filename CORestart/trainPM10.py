import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

def train_KNN(xTrain, yTrain):
    model = KNeighborsRegressor(metric="manhattan", n_neighbors=20, p=1, weights="uniform")
    model.fit(xTrain, yTrain)
    return model

def train_DT(xTrain, yTrain):
    model = DecisionTreeRegressor(criterion="absolute_error", max_depth=5, min_samples_leaf=4, min_samples_split=10, splitter="best")
    model.fit(xTrain, yTrain)
    return model


def main():
    xTrain = pd.read_csv("xTrain.csv")
    yTrain = pd.read_csv("yTrain.csv")
    model = train_KNN(xTrain, yTrain)

    xTest = pd.read_csv("xTest.csv")
    y_pred = model.predict(xTest)

    yTest = pd.read_csv("yTest.csv")
    mse = mean_squared_error(yTest, y_pred)
    r2 = r2_score(yTest, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R squared: {r2}")



if __name__ == "__main__":
    main()