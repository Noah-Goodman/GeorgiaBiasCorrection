import argparse
import json
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Grid Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    resultDict: dict
        A Python dictionary with "MSE", "RMSE", "MAE", "R2", "Time"
    bestParams: dict
        A Python dictionary with the best parameters chosen by GridSearch
    """
    start = time.time()
    grid_search = GridSearchCV(estimator=clf, param_grid=pgrid, cv=5)
    grid_search.fit(xTrain, yTrain)
    best_model = grid_search.best_estimator_

    prediction = best_model.predict(xTest)

    mse = mean_squared_error(yTest, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yTest, prediction)
    r2 = r2_score(yTest, prediction)

    timeElapsed = time.time() - start
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Time": timeElapsed
    }, grid_search.best_params_


def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Random Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset. The random search cv should try
    at most 33% of the possible combinations.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are lists of the fpr and tpr associated
        with different thresholds. You should be able to use this
        to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    start = time.time()
    total_combinations = np.prod([len(v) for v in pgrid.values()])
    n_iter = max(1, int(total_combinations * 0.33))
    random_search = RandomizedSearchCV(estimator=clf, param_distributions=pgrid, cv=5, n_iter=n_iter, random_state=42)

    random_search.fit(xTrain, yTrain)
    best_model = random_search.best_estimator_

    prediction = best_model.predict(xTest)

    mse = mean_squared_error(yTest, prediction)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(yTest, prediction)
    r2 = r2_score(yTest, prediction)

    timeElapsed = time.time() - start
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "Time": timeElapsed
    }, random_search.best_params_


def eval_searchcv(clfName, clf, clfGrid,
                  xTrain, yTrain, xTest, yTest,
                  perfDict, bestParamDict):
    # evaluate grid search and add to perfDict
    cls_perf, gs_p  = eval_gridsearch(clf, clfGrid, xTrain,
                                               yTrain, xTest, yTest)
    perfDict[clfName + " (Grid)"] = cls_perf

    # evaluate random search and add to perfDict
    clfr_perf, rs_p  = eval_randomsearch(clf, clfGrid, xTrain,
                                            yTrain, xTest, yTest)
    perfDict[clfName + " (Random)"] = clfr_perf
    bestParamDict[clfName] = {"Grid": gs_p, "Random": rs_p}
    return perfDict, bestParamDict


def your_model():
    """
    Return an instance of the optimal model based on your
    model selection and assessment strategy. This can be any
    of the 6 models tested in the homework and should be a
    parameter combination you tested.
    """
    # return DecisionTreeClassifier(criterion="gini", splitter="random", min_samples_leaf=2, min_samples_split=20, max_depth=10, random_state=42)



def get_parameter_grid(mName):
    """
    Given a model name, return the parameter grid associated with it

    Parameters
    ----------
    mName : string
        name of the model (e.g., DT, KNN, LR (None))

    Returns
    -------
    pGrid: dict
        A Python dictionary with the appropriate parameters for the model.
        The dictionary should have at least 2 keys and each key should have
        at least 2 values to try.
    """
    if mName == "DT":
        return {
            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
            "splitter": ["best", "random"],
            "max_depth": [3, 5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4]
        }
    elif mName == "Linear":
        return {
            'fit_intercept': [True, False],
            'copy_X': [True, False],
        }
    elif mName == "Ridge":
        return {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 2000, 5000, None],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag', 'saga'],
            'fit_intercept': [True, False]
        }
    elif mName == "Lasso":
        return {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'max_iter': [1000, 2000, 5000, None],
            'fit_intercept': [True, False]
        }
    elif mName == "KNN":
        return {
            'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2, 3, 4, 5]
        }
    # elif mName == "NN":
    #     return {
    #         'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    #         'activation': ['relu', 'tanh'],
    #         'alpha': [0.0001, 0.001, 0.01],
    #         'learning_rate': ['constant', 'adaptive'],
    #         'max_iter': [500]
    #     }
    return {}


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="yTest.csv",
                        help="filename for labels associated with the test data")
    # parser.add_argument("rocOutput",
    #                      help="csv filename for ROC curves")
    parser.add_argument("bestParamOutput",
                         help="json filename for best parameter")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    perfDict = {}
    bestParamDict = {}

    print("Tuning Decision Tree --------")
    # Compare Decision Tree
    dtName = "DT"
    dtGrid = get_parameter_grid(dtName)
    # fill in
    dtClf = DecisionTreeRegressor(random_state=42)
    perfDict, bestParamDict = eval_searchcv(dtName, dtClf, dtGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    print("Tuning Linear Regression --------")
    # logistic regression (unregularized)
    linearName = "Linear"
    unregLrGrid = get_parameter_grid(linearName)
    # fill in
    lrClf = LinearRegression()
    perfDict, bestParamDict = eval_searchcv(linearName, lrClf, unregLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    # logistic regression (L1)
    print("Tuning Ridge Regression --------")
    ridgeName = "Ridge"
    ridgeGrid = get_parameter_grid(ridgeName)
    ridgeClf = Ridge(random_state=42)
    perfDict, bestParamDict = eval_searchcv(ridgeName, ridgeClf, ridgeGrid,
                                           xTrain, yTrain, xTest, yTest,
                                           perfDict, bestParamDict)
    
    # logistic regression (L2)
    print("Tuning Lasso Regression --------")
    lassoName = "Lasso"
    lassoGrid = get_parameter_grid(lassoName)
    lassoClf = Lasso(random_state=42)
    perfDict, bestParamDict = eval_searchcv(lassoName, lassoClf, lassoGrid,
                                           xTrain, yTrain, xTest, yTest,
                                           perfDict, bestParamDict)
    # k-nearest neighbors
    print("Tuning K-nearest neighbors --------")
    knnName = "KNN"
    knnGrid = get_parameter_grid(knnName)
    # fill in
    knnClf = KNeighborsRegressor()
    perfDict, bestParamDict = eval_searchcv(knnName, knnClf, knnGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, bestParamDict)
    # neural networks
    # print("Tuning neural networks --------")
    # nnName = "NN"
    # nnGrid = get_parameter_grid(nnName)
    # # fill in
    # nnClf = MLPRegressor(random_state=42)
    # perfDict, bestParamDict = eval_searchcv(nnName, nnClf, nnGrid,
    #                                                xTrain, yTrain, xTest, yTest,
    #                                                perfDict, bestParamDict)
    perfDF = pd.DataFrame.from_dict(perfDict, orient='index')
    print(perfDF)
    # save roc curves to data.to_csv(args.rocOutput, index=False)
    # store the best parameters
    with open(args.bestParamOutput, 'w') as f:
        json.dump(bestParamDict, f)


if __name__ == "__main__":
    main()
