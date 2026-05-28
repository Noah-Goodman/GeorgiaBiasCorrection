import argparse
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def train_model(xTrain: pd.DataFrame, yTrain: pd.DataFrame, species: str):
    """
    Train a model given training data and model type
    
    Parameters
    ----------
    xTrain : pandas df with shape (n, d)
        Training features
    yTrain : pandas df with shape (n, )
        Training target
    species: str
        Species to train model for

    Returns
    -------
    model: trained model
    """
    if species == "no2":
        model = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
        model.fit(xTrain, yTrain)
    elif species == "no":
        model = KNeighborsRegressor(weights='distance', n_neighbors=11, metric='manhattan')
        model.fit(xTrain, yTrain)
    elif species == "o3":
        model = Ridge(alpha=1.0, fit_intercept=True, random_state=42)
        model.fit(xTrain, yTrain)
    elif species == "co":
        model = KNeighborsRegressor(weights='uniform', n_neighbors=15, p=2, metric='manhattan')
        model.fit(xTrain, yTrain)
    elif species == "pm10":
        model = LinearRegression(fit_intercept=False)
        model.fit(xTrain, yTrain)
    elif species == "pm25":
        model = DecisionTreeRegressor(criterion="absolute_error", min_samples_leaf=4, min_samples_split=2, max_depth=20, splitter="best", random_state=42)
        model.fit(xTrain, yTrain)
    return model

def predict_model(model, xTest: pd.DataFrame):
    """
    Make predictions using trained model
    
    Parameters
    ----------
    model : trained model
    xTest : pandas df with shape (m, d)
        Test features

    Returns
    -------
    yPred: np.array with shape (m, )
        Predictions
    """
    yPred = model.predict(xTest)
    return yPred

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('species', type=str,
                        help='Species to train model for (pm25, pm10, no2, no, o3, co)')
    args = parser.parse_args()
    xTrain = pd.read_csv(f"../.data/Pittsburgh/split/xTrain-{args.species}.csv")
    yTrain = pd.read_csv(f"../.data/Pittsburgh/split/yTrain-{args.species}.csv")
    model = train_model(xTrain, yTrain, args.species)
    joblib.dump(model, f"models/{args.species}-model.pkl")

    xTest = pd.read_csv(f"../.data/Pittsburgh/split/xTest-{args.species}.csv")
    yTest = pd.read_csv(f"../.data/Pittsburgh/split/yTest-{args.species}.csv")
    yPred = predict_model(model, xTest)
    mse = mean_squared_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"Test MSE for {args.species}: {mse}")
    print(f"Test R2 for {args.species}: {r2}")

if __name__ == "__main__":
    main()