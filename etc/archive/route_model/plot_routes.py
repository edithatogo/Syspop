from glob import glob
from os import makedirs
from os.path import exists, join
from pickle import load as pickle_load

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt
from PIL import Image


def get_minmax_latlon(coordinates: list) -> tuple[float, float, float, float]:
    """
    Calculates the minimum and maximum latitude and longitude from a list of coordinates.

    Args:
        coordinates (list): A list of (latitude, longitude) tuples.

    Returns:
        tuple[float, float, float, float]: A tuple containing
            (min_latitude, max_latitude, min_longitude, max_longitude).
    """
    # Get the min and max of lat and lon
    min_lat = min(coordinates, key=lambda x: x[0])[0]
    max_lat = max(coordinates, key=lambda x: x[0])[0]
    min_lon = min(coordinates, key=lambda x: x[1])[1]
    max_lon = max(coordinates, key=lambda x: x[1])[1]
    return min_lat, max_lat, min_lon, max_lon


def convert_png_to_gif(png_files: list[str], gif_path: str, duration: int):
    """
    Converts a list of PNG image files into a GIF.

    Args:
        png_files (list[str]): A list of filepaths to PNG images.
        gif_path (str): The filepath to save the output GIF.
        duration (int): The duration (in milliseconds) for each frame in the GIF.
    """
    # Read all png images into a list
    images = [Image.open(png_file) for png_file in png_files]

    # Save the images as a GIF
    images[0].save(
        gif_path, save_all=True, append_images=images[1:], duration=duration, loop=0
    )


def plot_data(all_data_points: list[tuple[float, float]], ax):
    """
    Plots a collection of data points (latitude, longitude) on a given Matplotlib Axes object.

    Args:
        all_data_points (list[tuple[float, float]]): A list of (latitude, longitude) tuples.
        ax (matplotlib.axes.Axes): The Matplotlib Axes object to plot on.
            It is assumed to have a Cartopy projection.
    """

    latitudes, longitudes = zip(*all_data_points)

    alpha_flag = 0.5
    color = "r"
    markersize = 5
    ax.plot(
        longitudes,
        latitudes,
        f"{color}o",
        markersize=markersize,
        alpha=alpha_flag,
        transform=ccrs.PlateCarree(),
    )


if __name__ == "__main__":

    workdir = "/home/zhangs/Github/Syspop/etc/route_model/agents_movement_output_v2.0"
    workdir_vis = join(workdir, "vis")
    plot_hours = [7, 8, 9]  # range(24)
    frames = 60

    if not exists(workdir_vis):
        makedirs(workdir_vis)

    all_files = glob(join(workdir, "routes_*.pickle"))

    all_data = {}

    for proc_hr in range(24):
        all_data[proc_hr] = []

    use_data_latlon_range = True

    if use_data_latlon_range:
        min_lat = 9999
        max_lat = -9999
        min_lon = 9999
        max_lon = -9999
    else:
        min_lat = -41.31
        max_lat = -41.26
        min_lon = 174.74
        max_lon = 174.79

    for proc_file in all_files:
        proc_data_all_agents = pickle_load(open(proc_file, "rb"))

        for proc_hr in range(24):

            for agent_id in proc_data_all_agents:

                proc_data = proc_data_all_agents[agent_id]

                if proc_hr not in proc_data:
                    continue

                if proc_data[proc_hr]["routes"] is None:
                    continue

                if use_data_latlon_range:
                    if len(proc_data[proc_hr]["routes"]) > 1:
                        proc_routes = proc_data[proc_hr]["routes"]
                        min_lat1, max_lat1, min_lon1, max_lon1 = get_minmax_latlon(
                            proc_routes
                        )
                        if min_lat1 < min_lat:
                            min_lat = min_lat1
                        if max_lat1 > max_lat:
                            max_lat = max_lat1
                        if min_lon1 < min_lon:
                            min_lon = min_lon1
                        if max_lon1 > max_lon:
                            max_lon = max_lon1

                all_data[proc_hr].append(proc_data[proc_hr]["routes"])

    # cimgt.Stamen.get_image = image_spoof  # reformat web request for street map spoofing
    request = cimgt.GoogleTiles(cache=True)  # spoofed, downloaded street map
    # request = cimgt.OSM()

    files_list_all = []
    files_list_hr = []
    for proc_hr in plot_hours:
        proc_data = all_data[proc_hr]
        files_list_hr = []
        if len(proc_data) == 0:
            continue

        for frame in range(frames):

            print(f"Hour: {proc_hr}; Frame: {frame}")

            fig, ax = plt.subplots(
                figsize=(10, 10), subplot_kw=dict(projection=request.crs)
            )
            ax.set_extent([min_lon, max_lon, min_lat, max_lat])
            ax.add_image(request, 12)
            ax.set_title(f"{str(proc_hr).zfill(2)} hr + {str(frame).zfill(2)} min")

            all_data_points = []
            for data_to_plot in proc_data:
                if len(data_to_plot) == 1:
                    data_to_plot2 = data_to_plot[0]
                else:
                    data_to_plot2 = data_to_plot[frame]

                all_data_points.append(data_to_plot2)

            plot_data(all_data_points, ax)

            proc_filepath = join(workdir_vis, f"test_{proc_hr}_{frame}.png")
            plt.savefig(proc_filepath, bbox_inches="tight")
            plt.close()
            files_list_all.append(proc_filepath)
            files_list_hr.append(proc_filepath)

        convert_png_to_gif(files_list_hr, join(workdir_vis, f"test_{proc_hr}.gif"), 200)

    convert_png_to_gif(files_list_all, join(workdir_vis, "test_all.gif"), 150)
