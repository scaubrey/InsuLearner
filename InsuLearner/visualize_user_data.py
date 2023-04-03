
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_style("darkgrid")


def plot_raw_data(user, start_date, end_date):
    """
    Plot Tidepool data in its original form, ie 5-min cgm time resolution

    Args:
        user: Tidepool_User
        start_date (dt.DateTime): start date to plot
        end_date (dt.DateTime): end date to plot
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    event_times = []
    cgm_values = []

    if len(user.cgm_timeline) > 0:

        for dt, cgm_event in user.cgm_timeline.items():
            if start_date <= dt <= end_date:
                event_times.append(dt)
                cgm_values.append(cgm_event.get_value())

        ax[0].scatter(event_times, cgm_values)
        ax[0].set_title("CGM")
        ax[0].set_ylabel("mg/dL")

    if len(user.note_timeline) > 0:
        note_times = []
        note_values = []
        for dt, note_event in user.note_timeline.items():
            if start_date <= dt <= end_date:
                note_times.append(dt)
                note_values.append(max(cgm_values))
        ax[0].plot(note_times, note_values, label="Notes", linestyle="None", marker="s", color="y")

    ax[0].legend()

    if len(user.bolus_timeline) > 0:

        event_times = []
        bolus_values = []

        for dt, bolus_event in user.bolus_timeline.items():
            if start_date <= dt <= end_date:
                event_times.append(dt)
                bolus_values.append(bolus_event.get_value())

        if len(bolus_values) > 0:
            ax[1].set_title("Bolus")
            ax[1].stem(event_times, bolus_values)
            ax[1].set_ylabel("Units")

    if len(user.food_timeline) > 0:

        event_times = []
        carb_values = []

        for dt, carb_event in user.food_timeline.items():
            if start_date <= dt <= end_date:
                event_times.append(dt)
                carb_values.append(carb_event.get_value())

        if len(carb_values) > 0:
            ax[2].stem(event_times, carb_values)
            ax[2].set_title("Carbs")
            ax[2].set_ylabel("Grams")

    plt.title(start_date.strftime("%Y-%m-%d"))
    plt.show()


def plot_daily_stats(daily_df):
    """
    Make a plot of daily info.

    Args:
        aggregated_df pd.DataFrame: rows are days and columns are stats
    """

    stats_to_plot = [
        "cgm_geo_mean",
        "cgm_geo_std",
        "total_insulin",
        "total_carbs",
        "carb_insulin_ratio",
        "cgm_mean_insulin_ratio",
        "cgm_mean_insulin_factor"
    ]

    fig, ax = plt.subplots(len(stats_to_plot), 1, figsize=(8, 10))

    for i, stat in enumerate(stats_to_plot):
        ax[i].bar(daily_df["date"], daily_df[stat])
        ax[i].set_title(stat)

    plt.show()


def plot_daily_stats_distribution(all_daily_df):
    stats_to_plot = [
        "cgm_geo_mean",
        "cgm_geo_std",
        # "total_insulin",
        # "total_carbs",
        # "carb_insulin_ratio",
        "cgm_geo_mean_norm",
        # "total_insulin_norm",
        # "total_carbs_norm",
        # "insulin_needs"
    ]

    fig, axs = plt.subplots(nrows=len(stats_to_plot), figsize=(10, 15))

    for i, stat in enumerate(stats_to_plot):
        sns.lineplot(x="date_idx", y=stat, data=all_daily_df, hue="user_id", ax=axs[i])

    plt.show()