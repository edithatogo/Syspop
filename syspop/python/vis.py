from collections import Counter
from os.path import join

from folium import Map as folium_map
from folium import PolyLine as folium_polyline
from folium.plugins import HeatMap
from matplotlib.pyplot import (
    axvline,
    bar,
    clim,
    close,
    colorbar,
    legend,
    plot,
    savefig,
    scatter,
    subplots,
    title,
    xlabel,
    xlim,
    xticks,
    ylabel,
    ylim,
)
from numpy import arange as numpy_arange
from numpy import polyfit as numpy_polyfit
from pandas import DataFrame, merge
from seaborn import histplot as sns_hisplot


def validate_vis_movement(
    val_dir: str,
    model_data: DataFrame,
    truth_data: DataFrame,
    merge_method: str = "inner",  # left, inner
    apply_factor: bool = False,
):
    """
    Visualizes and compares modeled commute movement data against truth data (e.g., census).

    Generates two scatter plots: one for the synthetic population's commute patterns
    and one for the truth data's commute patterns. Points are plotted based on
    home and work SA2 areas, colored and sized by the total number of commuters.

    Args:
        val_dir (str): Directory to save the output PNG plots.
        model_data (DataFrame): DataFrame of modeled commute data. Expected columns:
            'area_home', 'area_work', 'total' (number of commuters).
        truth_data (DataFrame): DataFrame of actual commute data (e.g., from census).
            Expected columns: 'area_home', 'area_work', 'total'.
        merge_method (str, optional): Method for merging model and truth data
            (e.g., "inner", "left"). Used to align data for comparison, though
            the direct comparison isn't plotted here, only separate visualizations.
            Defaults to "inner".
        apply_factor (bool, optional): If True, scales the 'total' in `truth_data`
            by the ratio of total commuters in `model_data` to `truth_data`.
            Defaults to False.
    """

    x = model_data[model_data["total"] > 10]
    y = truth_data[truth_data["total"] > 10]
    x = x[x["area_home"] != x["area_work"]]
    y = y[y["area_home"] != y["area_work"]]

    merged_df = merge(
        x, y, on=["area_home", "area_work"], suffixes=("_x", "_y"), how=merge_method
    )

    if len(merged_df) == 0:
        return

    if apply_factor:
        factor = merged_df["total_x"].sum() / merged_df["total_y"].sum()
        merged_df["total_y"] = merged_df["total_y"] * factor

    min_value = min(merged_df[["area_home", "area_work"]].min())
    max_value = min(merged_df[["area_home", "area_work"]].max())

    plot([min_value, max_value], [min_value, max_value], "k")

    scatter(
        merged_df["area_home"],
        merged_df["area_work"],
        c=merged_df["total_x"],
        s=merged_df["total_x"],
        cmap="jet",
        alpha=0.5,
    )
    title("Synthetic population")
    colorbar()
    xlim(min_value - 1000, max_value + 1000)
    ylim(min_value - 1000, max_value + 1000)
    clim([50, 450])
    xlabel("SA2")
    ylabel("SA2")
    savefig(join(val_dir, "validation_work_commute_pop.png"), bbox_inches="tight")
    close()

    plot([min_value, max_value], [min_value, max_value], "k")
    scatter(
        merged_df["area_home"],
        merged_df["area_work"],
        c=merged_df["total_y"],
        s=merged_df["total_y"],
        cmap="jet",
        alpha=0.5,
    )
    title("Census 2018")
    colorbar()
    xlim(min_value - 1000, max_value + 1000)
    ylim(min_value - 1000, max_value + 1000)
    clim([50, 450])
    xlabel("SA2")
    ylabel("SA2")
    savefig(join(val_dir, "validation_work_commute_census.png"), bbox_inches="tight")
    close()


def validate_vis_plot(
    output_dir: str,
    err_data: dict,
    data_title: str,
    output_filename: str,
    x_label: str = None,
    y_label: str | None = None,
    plot_ratio: bool = True,
):
    """
    Generates and saves a plot for validation purposes.

    If `plot_ratio` is True, it plots the values from `err_data` directly
    (assuming `err_data` is a dictionary or similar iterable of y-values).
    If `plot_ratio` is False, it expects `err_data` to be a dictionary
    with "truth" and "model" keys, and plots these two series against each other.

    Args:
        output_dir (str): Directory to save the output PNG plot.
        err_data (dict | list | array-like): Data to plot. Structure depends on `plot_ratio`.
        data_title (str): Title for the plot.
        output_filename (str): Base name for the output PNG file (without .png).
        x_label (str | None, optional): Label for the x-axis. Defaults to None.
        y_label (str | None, optional): Label for the y-axis. Defaults to None.
        plot_ratio (bool, optional): If True, plots `err_data` directly. If False,
            plots "truth" vs "model" from `err_data`. Defaults to True.
    """
    if plot_ratio:
        plot(err_data, "k.")
    else:
        plot(err_data["truth"], "b.", label="truth", alpha=0.5)
        plot(err_data["model"], "r.", label="model", alpha=0.5)
        legend()
    title(data_title)
    xlabel(x_label)
    ylabel(y_label)
    savefig(join(output_dir, f"{output_filename}.png"), bbox_inches="tight")
    close()


def validate_vis_barh(
    output_dir: str,
    err_data: dict,
    data_title: str,
    output_filename: str,
    x_label: str | None = None,
    y_label: str | None = None,
    plot_ratio: bool = True,
    add_polyfit: bool = False,
    figure_size: tuple | None = None,
):
    """
    Generates and saves a horizontal bar chart for validation purposes.

    If `plot_ratio` is True, it plots error percentages from `err_data` (a dictionary
    of category: error_value).
    If `plot_ratio` is False, it plots "truth" vs "model" values from `err_data`
    (a dictionary with "truth" and "model" keys, each holding category: value dicts).
    Optionally, it can add polynomial fit lines to the truth/model bars.

    Args:
        output_dir (str): Directory to save the output PNG plot.
        err_data (dict): Data to plot. Structure depends on `plot_ratio`.
        data_title (str): Title for the plot.
        output_filename (str): Base name for the output PNG file (without .png).
        x_label (str | None, optional): Label for the x-axis. Defaults to None.
        y_label (str | None, optional): Label for the y-axis (typically category names).
                                     Defaults to None.
        plot_ratio (bool, optional): If True, plots error ratios. If False, plots
            truth vs model values. Defaults to True.
        add_polyfit (bool, optional): If True and `plot_ratio` is False, adds a
            3rd-degree polynomial fit line for truth and model bars. Defaults to False.
        figure_size (tuple | None, optional): Tuple specifying (width, height) of
            the figure. Defaults to None (Matplotlib default).
    """
    # Create figure and axes
    if figure_size is None:
        fig, ax = subplots()
    else:
        fig, ax = subplots(figsize=figure_size)
    fig.tight_layout()

    # Set bar width
    bar_width = 0.35

    if plot_ratio:
        # Get keys and values
        keys = err_data.keys()
        x_vals = err_data.values()

        # Arrange keys on x-axis
        index = range(len(keys))
        ax.set_yticks(index)

        # Create bars for 'x' and 'y'
        ax.barh(index, list(x_vals), bar_width, color="b", label="Error percentage")
        axvline(x=0, color="red", linestyle="--", linewidth=2)
    else:
        keys = list(err_data["truth"].keys())
        truth_values = list(err_data["truth"].values())
        model_values = list(err_data["model"].values())

        # Create an array with the positions of each bar along the y-axis
        y_pos = numpy_arange(len(keys))

        # Create a horizontal bar chart
        ax.barh(y_pos - 0.2, truth_values, 0.4, color="blue", label="truth")
        ax.barh(y_pos + 0.2, model_values, 0.4, color="red", label="model")

        ax.set_yticks(y_pos, keys)

        if add_polyfit:
            # Fit a line to the truth values
            truth_fit = numpy_polyfit(y_pos, truth_values, 3)
            truth_fit_fn = numpy_polyfit(truth_fit)
            plot(truth_fit_fn(y_pos), y_pos, color="blue")

            # Fit a line to the model values
            model_fit = numpy_polyfit(y_pos, model_values, 3)
            model_fit_fn = numpy_polyfit(model_fit)
            plot(model_fit_fn(y_pos), y_pos, color="red")

    # Labeling
    if x_label is not None:
        ax.set_xlabel(x_label)

    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.set_title(f"{data_title}")
    ax.set_yticklabels(keys)
    ax.legend()
    savefig(join(output_dir, f"{output_filename}.png"), bbox_inches="tight")
    close()


def plot_pie_charts(output_dir: str, df: DataFrame):
    """
    Generates and saves pie charts for categorical columns and histograms
    for numerical columns in a DataFrame.

    For each column in the DataFrame:
    - If the column is numerical (int64, float64), a histogram with KDE is plotted.
    - If the column is categorical, a pie chart of value counts is plotted.
    Each plot is saved as a PNG file named after the column in `output_dir`.

    Args:
        output_dir (str): The directory where the output PNG plots will be saved.
        df (DataFrame): The DataFrame to analyze and plot.
    """
    for column in df.columns:
        if df[column].dtype in ["int64", "float64"]:
            sns_hisplot(df[column], kde=True)  # For numerical columns, use a histogram
        else:
            df[column].value_counts().plot(
                kind="pie"
            )  # For categorical columns, use a pie chart
        title(f"Distribution for {column}")
        savefig(join(output_dir, f"{column}.png"), bbox_inches="tight")
        close()


def plot_map_html(output_dir: str, df: DataFrame, data_name: str):
    """
    Generates and saves an HTML file containing a Folium map with a heatmap layer.

    The map is centered at the mean latitude and longitude of the input DataFrame.
    A heatmap is generated from the 'latitude' and 'longitude' points in the DataFrame.

    Args:
        output_dir (str): Directory to save the output HTML file.
        df (DataFrame): DataFrame containing 'latitude' and 'longitude' columns.
        data_name (str): Base name for the output HTML file (without .html).
    """
    # Create a map centered at an average location
    m = folium_map(
        location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=14
    )

    # Add a heatmap to the map
    HeatMap(data=df, radius=8, max_zoom=13).add_to(m)

    # Display the map
    m.save(join(output_dir, f"{data_name}.html"))


def plot_travel_html(output_dir: str, df: DataFrame, data_name: str):
    """
    Generates and saves an HTML file containing a Folium map displaying travel trips as lines.

    The map is centered at the mean start latitude and longitude of the trips.
    Each row in the DataFrame, representing a trip, is plotted as a PolyLine
    from ('start_lat', 'start_lon') to ('end_lat', 'end_lon').

    Args:
        output_dir (str): Directory to save the output HTML file.
        df (DataFrame): DataFrame of travel trips. Expected columns:
                        'start_lat', 'start_lon', 'end_lat', 'end_lon'.
        data_name (str): Base name for the output HTML file (without .html).
    """

    m = folium_map(
        location=[df["start_lat"].mean(), df["start_lon"].mean()],
        zoom_start=13,
        prefer_canvas=True,
    )

    for idx, row in df.iterrows():
        folium_polyline(
            [(row["start_lat"], row["start_lon"]), (row["end_lat"], row["end_lon"])],
            color="red",
            weight=2.5,
            opacity=1,
        ).add_to(m)

    # Display the map
    m.save(join(output_dir, f"{data_name}.html"))


def plot_location_timeseries_charts(output_dir: str, location_counts: dict[str, dict[int, int]]):
    """
    Generates and saves time series plots showing the number of people at
    different location types for each hour.

    For each location type (key in `location_counts`), a line plot is created
    showing the count of people at that location type over 24 hours.
    Each plot is saved as a PNG file named "{location_type}_distribution.png".

    Args:
        output_dir (str): Directory to save the output PNG plots.
        location_counts (dict[str, dict[int, int]]): A dictionary where outer keys
            are location type names (e.g., "home", "work") and inner dictionaries
            map integer hours (0-23) to the count of people at that location type
            during that hour.
    """

    for proc_loc in location_counts:
        proc_data = location_counts[proc_loc]
        # Create lists for the plot
        hours = list(proc_data.keys())
        people = list(proc_data.values())

        plot(hours, people, marker="o")  # Plot the data
        xticks(range(0, max(hours)))  # Set the x-ticks to be hourly
        xlabel("Hour")  # Set x-axis label
        ylabel("Number of People")  # Set y-axis label
        title(f"Number of People by Hour \n {proc_loc}")  # Set title
        savefig(join(output_dir, f"{proc_loc}_distribution.png"), bbox_inches="tight")
        close()


def plot_location_occurence_charts_by_hour(
    output_dir: str,
    location_counts: dict,
    hour: int,
    data_type: str,
):
    """
    Generates and saves a bar chart showing the distribution of the number of
    people at different specific locations of a given `data_type` at a specific `hour`.

    For example, if `data_type` is "supermarket" and `hour` is 10, this plots
    how many supermarkets have 0 people, 1 person, 2 people, etc., at 10:00.

    Args:
        output_dir (str): Directory to save the output PNG plot.
        location_counts (dict): A dictionary where keys are specific location IDs
                                (e.g., "supermarket_A", "supermarket_B") and values
                                are the count of people at that location at the given `hour`.
        hour (int): The specific hour for which the distribution is plotted.
        data_type (str): The generic type of location (e.g., "supermarket"), used
                         for titling and file naming.
    """
    counts = list(location_counts.values())

    counts_processed = Counter(counts)

    # Extract unique values and their counts
    values = list(counts_processed.keys())
    occurrences = list(counts_processed.values())

    # Plotting
    bar(values, occurrences)
    title(
        f"Number of People distribution \n Hour: {hour}, Data: {data_type}"
    )  # Set title
    xlabel("Number of people")
    ylabel(f"Number of places ({data_type})")
    savefig(join(output_dir, f"{data_type}_{hour}_hist.png"), bbox_inches="tight")
    close()


def plot_average_occurence_charts(output_dir: str, data_counts: list[float], data_type: str):
    """
    Generates and saves a line plot showing the average number of people per
    location of a specific `data_type` over time (presumably hourly).

    Args:
        output_dir (str): Directory to save the output PNG plot.
        data_counts (list[float]): A list where each element is the average
                                   number of people per location of `data_type`
                                   for a specific time step (e.g., hour).
        data_type (str): The generic type of location (e.g., "supermarket"),
                         used for titling and file naming.
    """

    plot(data_counts)
    title(f"Average number of people per {data_type}")
    xlabel("Hour")
    ylabel("Number of people")
    savefig(
        join(output_dir, f"average_{data_type}_timeseries.png"), bbox_inches="tight"
    )
    close()
