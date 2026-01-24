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
        model = Lasso(alpha=0.001, max_iter=5000, fit_intercept=False)
        model.fit(xTrain, yTrain)
    elif species == "no":
        model = KNeighborsRegressor(weights='distance', n_neighbors=7, p=2, metric='manhattan')
        model.fit(xTrain, yTrain)
    elif species == "o3":
        model = Lasso(alpha=0.001, max_iter=5000, fit_intercept=False)
        model.fit(xTrain, yTrain)
    elif species == "co":
        model = DecisionTreeRegressor(criterion="squared_error", min_samples_leaf=4, min_samples_split=10, max_depth=15, splitter="random", random_state=42)
        model.fit(xTrain, yTrain)
    elif species == "pm10":
        model = DecisionTreeRegressor(criterion="absolute_error", min_samples_leaf=4, min_samples_split=2, max_depth=15, splitter="random", random_state=42)
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
    xTrain = pd.read_csv(f"../.data/Chapel-Hill/split/xTrain-{args.species}.csv")
    yTrain = pd.read_csv(f"../.data/Chapel-Hill/split/yTrain-{args.species}.csv")
    model = train_model(xTrain, yTrain, args.species)
    joblib.dump(model, f"models/{args.species}-model.pkl")

    xTest = pd.read_csv(f"../.data/Chapel-Hill/split/xTest-{args.species}.csv")
    yTest = pd.read_csv(f"../.data/Chapel-Hill/split/yTest-{args.species}.csv")
    yPred = predict_model(model, xTest)
    mse = mean_squared_error(yTest, yPred)
    r2 = r2_score(yTest, yPred)
    print(f"Test MSE for {args.species}: {mse}")
    print(f"Test R2 for {args.species}: {r2}")

if __name__ == "__main__":
    main()