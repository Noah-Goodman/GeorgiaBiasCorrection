import argparse
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from mizani.breaks import breaks_date
from plotnine import (aes, element_blank, element_text, geom_label, geom_line,
                      geom_point, ggplot, labs, scale_color_manual,
                      scale_x_datetime, scale_y_continuous, theme, theme_bw)
import joblib

# Gaps longer than this many hours are drawn as a break instead of a straight line
MAX_GAP_HOURS = 50

# How many of the highest peaks to mark on each plot
N_PEAKS = 5

# Maxima closer together than this are treated as the same peak
PEAK_MIN_SEPARATION_HOURS = 24

# Line width and peak marker size, in plotnine's size units
LINE_SIZE = 0.75
PEAK_MARKER_SIZE = 3

# Peak labels sit this fraction of the y range above their dot
PEAK_LABEL_NUDGE = 0.03

# Display names with subscripted numbers (matplotlib mathtext)
SPECIES_LABELS = {
    "pm25": "PM$_{2.5}$",
    "pm10": "PM$_{10}$",
    "no2": "NO$_2$",
    "o3": "O$_3$",
}

# Color each series is drawn in, keyed by the label it carries in the legend
SERIES_COLORS = {
    "FRM/FEM": "#FF6600",
    "Corrected": "blue",
    "Raw": "green",
}


def species_label(species: str) -> str:
    """Species name for display, with numbers subscripted"""
    return SPECIES_LABELS.get(species, species.upper())


def gap_segments(times, values, series: str, max_gap_hours: float = MAX_GAP_HOURS):
    """
    Put one series in long form, split into segments that are drawn separately

    Parameters
    ----------
    times : sequence of timestamps with shape (m, )
        x values, converted to datetime and sorted
    values : sequence with shape (m, )
        y values
    series : str
        Name this series carries in the legend
    max_gap_hours : float
        Any gap between consecutive timestamps longer than this is broken

    Returns
    -------
    Long form df with columns dayhour, value, series and segment. geom_line groups
    on segment, so no line is drawn across a large gap
    """

    df = pd.DataFrame({
        "dayhour": pd.to_datetime(pd.Series(np.asarray(times))),
        "value": pd.Series(np.asarray(values, dtype=float)),
    }).sort_values("dayhour", kind="stable").reset_index(drop=True)

    gaps = df["dayhour"].diff() / pd.Timedelta(hours=1)

    # A long gap or a missing sample both start a new segment, so the line lifts there
    starts = (gaps > max_gap_hours) | df["value"].shift().isna()

    df["series"] = series
    df["segment"] = series + "-" + starts.cumsum().astype(str)
    return df.dropna(subset=["value"]).reset_index(drop=True)


def annotate_peaks(df: pd.DataFrame, color: str, nudge: float, n_peaks: int = N_PEAKS,
                   min_separation_hours: float = PEAK_MIN_SEPARATION_HOURS):
    """
    Layers marking the n highest peaks of a series with a dot labeled by its y value

    Parameters
    ----------
    df : long form df with columns dayhour and value
        One series, as returned by gap_segments
    color : str
        Color of the dots and labels, normally the color of the series
    nudge : float
        How far above its dot each label sits, in y units
    n_peaks : int
        How many peaks to mark
    min_separation_hours : float
        Maxima within this many hours of an already marked peak are skipped, so
        one tall spike does not use up every marker

    Returns
    -------
    List of layers to add to the plot, empty when there is nothing to mark
    """

    if n_peaks <= 0 or df.empty:
        return []

    times = df["dayhour"].to_numpy()
    values = df["value"].to_numpy(dtype=float)

    separation = np.timedelta64(int(min_separation_hours * 3600), "s")

    peaks = []
    for i in np.argsort(values)[::-1]:
        if len(peaks) >= n_peaks:
            break
        if all(abs(times[i] - times[j]) >= separation for j in peaks):
            peaks.append(i)

    if not peaks:
        return []

    marked = pd.DataFrame({
        "dayhour": times[peaks],
        "value": values[peaks],
        "label": [f"{values[i]:.1f}" for i in peaks],
    })

    return [
        geom_point(marked, aes("dayhour", "value"), color=color,
                   size=PEAK_MARKER_SIZE, inherit_aes=False),
        geom_label(marked, aes("dayhour", "value", label="label"), color=color,
                   fill="white", alpha=0.7, size=8, label_size=0, label_padding=0.2,
                   nudge_y=nudge, inherit_aes=False),
    ]


def base_theme():
    """Theme shared by every plot: white panel, major grid only, rotated date ticks"""

    return (theme_bw()
            + theme(figure_size=(12, 6),
                    panel_grid_minor=element_blank(),
                    axis_text_x=element_text(rotation=45, ha="right"),
                    plot_title=element_text(ha="center"),
                    legend_title=element_blank()))


def dayhour_split(species: str):
    preprocessed = pd.read_csv(f"../.data/Pittsburgh/collocation/preprocessed-{species}.csv")
    rng = np.random.default_rng(seed=25)
    N = len(preprocessed)
    permuted_indices = rng.permutation(N)
    preprocessed = preprocessed.iloc[permuted_indices]
    dayhour = preprocessed["dayhour"]
    split = int(N*0.8)
    train_dayhour = dayhour[:split]
    test_dayhour = dayhour[split:]
    return train_dayhour, test_dayhour

def plot_training(model, yTest: pd.DataFrame, xTest: pd.DataFrame, species: str,
                  max_gap_hours: float = MAX_GAP_HOURS, n_peaks: int = N_PEAKS):
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
    max_gap_hours : float
        Gaps longer than this are left blank instead of connected
    n_peaks : int
        How many of the highest corrected peaks to mark and label

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

    raw_QAQ = pd.read_csv(f"../.data/Pittsburgh/collocation/preprocessed-{species}.csv")
    raw_QAQ["dayhour"] = pd.to_datetime(raw_QAQ["dayhour"])

    # Break each series across large gaps so they are not connected by a straight line
    order = ["FRM/FEM", "Corrected", "Raw"]
    data = pd.concat([
        gap_segments(yTest["dayhour"], yTest[species], "FRM/FEM", max_gap_hours),
        gap_segments(yPrediction["dayhour"], yPrediction[species], "Corrected", max_gap_hours),
        gap_segments(raw_QAQ["dayhour"], raw_QAQ[species], "Raw", max_gap_hours),
    ], ignore_index=True)
    data["series"] = pd.Categorical(data["series"], categories=order, ordered=True)

    nudge = PEAK_LABEL_NUDGE * (data["value"].max() - data["value"].min())
    peaks = (annotate_peaks(data[data["series"] == "Corrected"],
                            SERIES_COLORS["Corrected"], nudge, n_peaks)
             + annotate_peaks(data[data["series"] == "Raw"],
                              SERIES_COLORS["Raw"], nudge, n_peaks))

    label = species_label(species)

    if species in ["pm25", "pm10"]:
        ylabel = f'{label} [µg/m³]'
    else:
        ylabel = f'{label} [ppb]'

    # Plain-text title for the filename, subscripted version for the figure
    title = f'Pittsburgh Hourly {label} After Correction'

    plot = (ggplot(data, aes("dayhour", "value", color="series", group="segment"))
            + geom_line(alpha=0.6, size=LINE_SIZE)
            + peaks
            + scale_color_manual(values=SERIES_COLORS, limits=order)
            + scale_x_datetime(breaks=breaks_date(10))
            # Leave room above the tallest peak for its label
            + scale_y_continuous(expand=(0.05, 0, 0.12 if peaks else 0.05, 0))
            + labs(x='Date', y=ylabel, title=title)
            + base_theme())

    plot.save(f"figures/{title}.png", dpi=100, verbose=False)

def plot_current(model, current: pd.DataFrame, species: str,
                 max_gap_hours: float = MAX_GAP_HOURS, n_peaks: int = N_PEAKS):
    """
    Plot current period results

    Parameters
    ----------
    model : trained model
    current : pandas df with shape (m, d)
        Current data with non features
    species: str
        Species to plot results for
    max_gap_hours : float
        Gaps longer than this are left blank instead of connected
    n_peaks : int
        How many of the highest corrected peaks to mark and label

    Returns
    -------
    None
    """

    df_dayhour = pd.to_datetime(current["dayhour"])
    current = current.drop(columns=["dayhour"])
    yPred = np.asarray(model.predict(current)).ravel()
    yPred = np.clip(yPred, 0, None)

    # add dayhour column back in
    current["dayhour"] = df_dayhour

    # Break each series across large gaps so they are not connected by a straight line
    order = ["Raw", "Corrected"]
    data = pd.concat([
        gap_segments(current["dayhour"], current[species], "Raw", max_gap_hours),
        gap_segments(current["dayhour"], yPred, "Corrected", max_gap_hours),
    ], ignore_index=True)
    data["series"] = pd.Categorical(data["series"], categories=order, ordered=True)

    nudge = PEAK_LABEL_NUDGE * (data["value"].max() - data["value"].min())
    peaks = (annotate_peaks(data[data["series"] == "Raw"],
                            SERIES_COLORS["Raw"], nudge, n_peaks)
             + annotate_peaks(data[data["series"] == "Corrected"],
                              SERIES_COLORS["Corrected"], nudge, n_peaks))

    label = species_label(species)

    if species in ["pm25", "pm10"]:
        ylabel = f'{label} [µg/m³]'
    else:
        ylabel = f'{label} [ppb]'

    # Plain-text title for the filename, subscripted version for the figure
    title = f'Pittsburgh Hourly {label}'

    plot = (ggplot(data, aes("dayhour", "value", color="series", group="segment"))
            + geom_line(alpha=0.6, size=LINE_SIZE)
            + peaks
            + scale_color_manual(values=SERIES_COLORS, limits=order)
            + scale_x_datetime(breaks=breaks_date(10))
            # Leave room above the tallest peak for its label
            + scale_y_continuous(expand=(0.05, 0, 0.12 if peaks else 0.05, 0))
            + labs(x='Date', y=ylabel, title=title)
            + base_theme())

    plot.save(f"figures/{title}.png", dpi=100, verbose=False)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('species', type=str,
                           help='Species to plot results for (pm25, pm10, no2, no, o3, co)')
    argparser.add_argument('-f', type=str, default='current',
                           help='Whether to plot train or current results')
    argparser.add_argument('--max-gap', type=float, default=MAX_GAP_HOURS,
                           help='Gaps longer than this many hours are left blank instead of connected')
    argparser.add_argument('--peaks', type=int, default=N_PEAKS,
                           help='How many of the highest corrected peaks to mark (0 to disable)')
    args = argparser.parse_args()
    model = joblib.load(f"models/{args.species}-model.pkl")

    if args.f == 'train':
        xTest = pd.read_csv(f"../.data/Pittsburgh/split/xTest-{args.species}.csv")
        yTest = pd.read_csv(f"../.data/Pittsburgh/split/yTest-{args.species}.csv")
        plot_training(model, yTest, xTest, args.species, args.max_gap, args.peaks)
    elif args.f == 'current':
        current = pd.read_csv(f"../.data/Pittsburgh/current/preprocessed-{args.species}.csv")
        plot_current(model, current, args.species, args.max_gap, args.peaks)

if __name__ == "__main__":
    main()
