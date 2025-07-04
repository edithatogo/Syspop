
from matplotlib.pyplot import close, gca, legend, savefig, title, xlabel, ylabel
from matplotlib.ticker import FuncFormatter
from pandas import DataFrame

from funcs import LOCATIONS_CFG


def plot_diary_percentage(
    data_to_plot: DataFrame,
    diary_vis_path: str,
    title_str: str | None = None,
    color_unknown: str = "#bfbfbd",
):
    """
    Generates and saves a stacked bar plot showing the percentage of different
    locations people are at for each hour of the day.

    Args:
        data_to_plot (DataFrame): DataFrame containing diary data with "Hour"
                                  and "Location" columns.
        diary_vis_path (str): Filepath to save the generated plot.
        title_str (str | None, optional): Optional additional title string to append
                                       to the default plot title. Defaults to None.
        color_unknown (str, optional): Hex color code for locations not found in
                                   LOCATIONS_CFG. Defaults to "#bfbfbd".
    """

    df_grouped = data_to_plot.groupby(["Hour", "Location"]).size().unstack(fill_value=0)
    df_percentage = df_grouped.divide(df_grouped.sum(axis=1), axis=0)

    colors = []
    columns = []
    for col in df_percentage.columns:
        if col not in LOCATIONS_CFG:
            colors.append(color_unknown)
        else:
            colors.append(LOCATIONS_CFG[col]["color"])
        columns.append(col)

    df_percentage[columns].plot(
        kind="bar",
        stacked=True,
        figsize=(10, 7),
        color=colors,
    )

    def to_percentage(y, _):
        return "{:.0%}".format(y)

    # Apply percentage formatting to y-axis ticks
    formatter = FuncFormatter(to_percentage)
    gca().yaxis.set_major_formatter(formatter)

    title_str_base = "Percentage of Different Locations for Each Hour"

    if title_str is not None:
        title_str_base += f" \n {title_str}"

    title(f"{title_str_base}")
    xlabel("Hour")
    ylabel("Percentage")
    legend(title="Location")

    savefig(
        diary_vis_path,
        bbox_inches="tight",
    )
    close()
