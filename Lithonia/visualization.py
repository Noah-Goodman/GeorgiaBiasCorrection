import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from mizani.breaks import breaks_date
from plotnine import (aes, element_blank, element_text, geom_line, ggplot, labs,
                      scale_color_manual, scale_x_datetime, theme, theme_bw)
import joblib

# Line width, in plotnine's size units
LINE_SIZE = 0.75


def long_form(times, values, series: str) -> pd.DataFrame:
    """
    Put one series in the long form plotnine draws from

    Parameters
    ----------
    times : sequence of timestamps with shape (m, )
        x values, converted to datetime
    values : sequence with shape (m, )
        y values
    series : str
        Name this series carries in the legend

    Returns
    -------
    Long form df with columns dayhour, value and series
    """

    return pd.DataFrame({
        "dayhour": pd.to_datetime(pd.Series(np.asarray(times))),
        "value": np.asarray(values, dtype=float).ravel(),
        "series": series,
    })


def base_theme():
    """Theme shared by every plot: white panel, major grid only, rotated date ticks"""

    return (theme_bw()
            + theme(figure_size=(12, 6),
                    panel_grid_minor=element_blank(),
                    axis_text_x=element_text(rotation=45, ha="right"),
                    plot_title=element_text(ha="center"),
                    legend_title=element_blank()))


def dayhour_split(species: str):
    preprocessed = pd.read_csv(f"../.data/Lithonia/collocation/preprocessed-{species}.csv")
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

    raw_QAQ = pd.read_csv(f"../.data/Lithonia/collocation/preprocessed-{species}.csv")
    raw_QAQ["dayhour"] = pd.to_datetime(raw_QAQ["dayhour"])
    if species == "pm10":
        raw_QAQ[species] = raw_QAQ[species].clip(upper=300)

    real = f"Real {species.upper()} Values"
    predicted = f"Predicted {species.upper()} Values"
    raw = f"Raw {species.upper()} Values"

    # Legend order follows the order the series are listed in, as plot order did
    order = [real, predicted, raw]
    colors = {real: "blue", predicted: "red", raw: "green"}

    data = pd.concat([
        long_form(yTest["dayhour"], yTest[species], real),
        long_form(yPrediction["dayhour"], yPrediction[species], predicted),
        long_form(raw_QAQ["dayhour"], raw_QAQ[species], raw),
    ], ignore_index=True)
    data["series"] = pd.Categorical(data["series"], categories=order, ordered=True)

    if species in ["pm25", "pm10"]:
        ylabel = f'{species.upper()} (µg/m³)'
    else:
        ylabel = f'{species.upper()} (PPB)'

    title = f'{species.upper()} Levels: Real vs Predicted vs Raw'

    plot = (ggplot(data, aes("dayhour", "value", color="series"))
            + geom_line(alpha=0.6, size=LINE_SIZE)
            + scale_color_manual(values=colors, limits=order)
            + scale_x_datetime(breaks=breaks_date(10))
            + labs(x='dayhour', y=ylabel, title=title)
            + base_theme())

    plot.save(f"figures/{title}.png", dpi=100, verbose=False)

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

    if species == "pm10":
        current[species] = current[species].clip(upper=300)

    predicted = f"Predicted {species.upper()} Values"
    raw = f"Raw {species.upper()} Values"

    # Legend order follows the order the series are listed in, as plot order did
    order = [raw, predicted]
    colors = {raw: "blue", predicted: "red"}

    data = pd.concat([
        long_form(current["dayhour"], current[species], raw),
        long_form(current["dayhour"], yPred, predicted),
    ], ignore_index=True)
    data["series"] = pd.Categorical(data["series"], categories=order, ordered=True)

    if species in ["pm25", "pm10"]:
        ylabel = f'{species.upper()} (µg/m³)'
    else:
        ylabel = f'{species.upper()} (PPB)'

    title = f'Current {species.upper()} Levels: Raw vs Predicted'

    plot = (ggplot(data, aes("dayhour", "value", color="series"))
            + geom_line(alpha=0.6, size=LINE_SIZE)
            + scale_color_manual(values=colors, limits=order)
            + scale_x_datetime(breaks=breaks_date(10))
            + labs(x='dayhour', y=ylabel, title=title)
            + base_theme())

    plot.save(f"figures/{title}.png", dpi=100, verbose=False)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('species', type=str,
                           help='Species to plot results for (pm25, pm10, no2, no, o3, co)')
    argparser.add_argument('-f', type=str, default='current',
                           help='Whether to plot train or current results')
    args = argparser.parse_args()
    model = joblib.load(f"models/{args.species}-model.pkl")

    if args.f == 'train':
        xTest = pd.read_csv(f"../.data/Lithonia/split/xTest-{args.species}.csv")
        yTest = pd.read_csv(f"../.data/Lithonia/split/yTest-{args.species}.csv")
        plot_training(model, yTest, xTest, args.species)
    elif args.f == 'current':
        current = pd.read_csv(f"../.data/Lithonia/current/preprocessed-{args.species}.csv")
        plot_current(model, current, args.species)

if __name__ == "__main__":
    main()
