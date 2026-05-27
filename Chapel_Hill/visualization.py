import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import joblib


def dayhour_split(species: str):
    preprocessed = pd.read_csv(f"../.data/Chapel-Hill/collocation/preprocessed-{species}.csv")
    rng = np.random.default_rng(seed=25)
    N = len(preprocessed)
    permuted_indices = rng.permutation(N)
    preprocessed = preprocessed.iloc[permuted_indices]
    dayhour = preprocessed["dayhour"]
    split = int(N*0.8)
    train_dayhour = dayhour[:split]
    test_dayhour = dayhour[split:]
    return train_dayhour, test_dayhour

def plot_training(model, yTest: pd.DataFrame, xTest: pd.DataFrame, species: str):
    """
    Plot training results
    
    Parameters
    ----------
    model : trained model
    yTest : pandas df with shape (m, )
        Test target
    xTest : pandas df with shape (m, d)
        Test features
    species: str
        Species to plot results for

    Returns
    -------
    None
    """

    yPred = model.predict(xTest).flatten()

    _train_dayhour, test_dayhour = dayhour_split(species)

    # Get the dayhours
    _train_dayhour, test_dayhour = dayhour_split(species)

    # Make sure test_dayhour matches the length of yTest
    test_dayhour = test_dayhour.reset_index(drop=True)
    yTest = yTest.reset_index(drop=True)

    # Add dayhour to dataframes
    yTest["dayhour"] = test_dayhour.values
    yPrediction = pd.DataFrame({
        "dayhour": test_dayhour.values,
        species: yPred
    })

    # Sort by dayhour for plotting
    yTest = yTest.sort_values("dayhour").reset_index(drop=True)
    yPrediction = yPrediction.sort_values("dayhour").reset_index(drop=True)

    # Convert all dayhour columns to datetime
    yTest["dayhour"] = pd.to_datetime(yTest["dayhour"])
    yPrediction["dayhour"] = pd.to_datetime(yPrediction["dayhour"])

    raw_QAQ = pd.read_csv(f"../.data/Chapel-Hill/collocation/preprocessed-{species}.csv")
    raw_QAQ["dayhour"] = pd.to_datetime(raw_QAQ["dayhour"])
    plt.figure(figsize=(12, 6))
    plt.plot(yTest["dayhour"], yTest[species], label=f"Real {species.upper()} Values", color='blue', alpha=0.6)
    plt.plot(yPrediction["dayhour"], yPrediction[species], label=f"Predicted {species.upper()} Values", color='red', alpha=0.6)
    plt.plot(raw_QAQ["dayhour"], raw_QAQ[species], label=f"Raw {species.upper()} Values", color='green', alpha=0.6)
    plt.xlabel('dayhour')
    if species in ["pm25", "pm10"]:
        plt.ylabel(f'{species.upper()} (µg/m³)')
    else:
        plt.ylabel(f'{species.upper()} (PPB)')
    plt.legend()

    title = f'{species.upper()} Levels: Real vs Predicted vs Raw'
    plt.title(title)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/{title}.png")

def plot_current(model, current: pd.DataFrame, species: str):
    """
    Plot current period results
    
    Parameters
    ----------
    model : trained model
    current : pandas df with shape (m, d)
        Current data with non features
    species: str
        Species to plot results for

    Returns
    -------
    None
    """

    df_dayhour = current["dayhour"]
    current = current.drop(columns=["dayhour"])
    yPred = model.predict(current)
    yPred = np.clip(yPred, 0, None)

    # add dayhour column back in
    current["dayhour"] = df_dayhour

    plt.figure(figsize=(12, 6))
    plt.plot(current["dayhour"], current[species], label=f"Raw {species.upper()} Values", color='blue', alpha=0.6)
    plt.plot(current["dayhour"], yPred, label=f"Predicted {species.upper()} Values", color='red', alpha=0.6)
    plt.xlabel('dayhour')
    if species in ["pm25", "pm10"]:
        plt.ylabel(f'{species.upper()} (µg/m³)')
    else:
        plt.ylabel(f'{species.upper()} (PPB)')
    plt.legend()

    title = f'Current {species.upper()} Levels: Raw vs Predicted'
    plt.title(title)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"figures/{title}.png")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('species', type=str,
                           help='Species to plot results for (pm25, pm10, no2, no, o3, co)')
    argparser.add_argument('-f', type=str, default='current',
                           help='Whether to plot train or current results')
    args = argparser.parse_args()
    model = joblib.load(f"models/{args.species}-model.pkl")

    if args.f == 'train':
        xTest = pd.read_csv(f"../.data/Chapel-Hill/split/xTest-{args.species}.csv")
        yTest = pd.read_csv(f"../.data/Chapel-Hill/split/yTest-{args.species}.csv")
        plot_training(model, yTest, xTest, args.species)
    elif args.f == 'current':
        current = pd.read_csv(f"../.data/Chapel-Hill/current/preprocessed-{args.species}.csv")
        plot_current(model, current, args.species)

if __name__ == "__main__":
    main()