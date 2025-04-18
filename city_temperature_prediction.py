import os
import numpy as np
import tools
from polynomial_fitting import PolynomialFitting
from matplotlib import pyplot as plt
import pandas as pd

PLOTS_DIR = "temprature_plots"
TEMP_CSV_PATH = "city_temperature.csv"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df = df[df["Temp"] > -30]  # seems like temp ~-72 is the mark of a missing value
    return df


def explore_israel_temperature(df: pd.DataFrame) -> None:
    """
    Plot temperature in Israel over the year for each recorded year.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed city temperature dataset (with DayOfYear column)
    """
    israel_df = df[df["Country"] == "Israel"]

    tools.config_plot(
        title="Daily Temperature in Israel by Year",
        xlabel="Day of Year",
        ylabel="Temperature (°C)",
        figsize=(10, 5),
    )
    for year, group in israel_df.groupby("Year"):
        plt.scatter(group["DayOfYear"], group["Temp"], s=10, label=year, alpha=0.5)
    plt.legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    tools.save_and_done_plot(os.path.join(PLOTS_DIR, "daily_temp_by_year.png"))

    std_by_month = israel_df.groupby("Month")["Temp"].std()
    tools.config_plot(
        title="Standard Deviation of Temperature by Month in Israel",
        xlabel="Month",
        ylabel="Standard Deviation (°C)",
        figsize=(10, 5),
    )
    plt.bar(std_by_month.index, std_by_month.values, color="skyblue", edgecolor="black")
    plt.xticks(std_by_month.index)
    tools.save_and_done_plot(os.path.join(PLOTS_DIR, "std_by_month.png"))


def explore_differences_between_countries(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby(["Country", "Month"])["Temp"].agg(["mean", "std"]).reset_index()
    )
    countries = grouped["Country"].unique()
    tools.config_plot(
        title="Average Monthly Temperature by Country",
        xlabel="Month",
        ylabel="Temperature (°C)",
        figsize=(10, 6),
    )
    for country in countries:
        country_data = grouped[grouped["Country"] == country]
        plt.errorbar(
            country_data["Month"],
            country_data["mean"],
            yerr=country_data["std"],
            label=country,
            capsize=3,
            elinewidth=2,
            marker="o",
            markersize=5,
            linestyle="-",
            alpha=0.7,
        )
    plt.xticks(range(1, 13))
    plt.legend()
    plt.grid(True)
    tools.save_and_done_plot(os.path.join(PLOTS_DIR, "avg_temp_by_country.png"))


def fit_model_for_different_k(df: pd.DataFrame) -> None:
    """
    Fit a polynomial regression model for different values of k
    """
    israel_df = df[df["Country"] == "Israel"]
    israel_df = israel_df.drop(["Country", "City", "Date"], axis=1)
    X, y = israel_df.drop("Temp", axis=1), israel_df.Temp
    # take only israel data
    X_train, y_train, X_test, y_test = tools.train_test_split(X, y, test_size=0.25)
    X_train = X_train["DayOfYear"].values
    X_test = X_test["DayOfYear"].values
    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k=k)
        model.fit(X_train, y_train)
        # calculate loss to 2 decimal places
        loss = np.round(model.loss(X_test, y_test), 2)
        losses.append(loss)
        print(f"Loss for k={k}: {loss}")

    # plot loss vs k
    tools.config_plot(title="Loss vs k", xlabel="k", ylabel="Loss", figsize=(10, 5))
    plt.bar(range(1, 11), losses, color="skyblue", edgecolor="black")
    plt.xticks(range(1, 11))
    tools.save_and_done_plot(os.path.join(PLOTS_DIR, "loss_vs_k.png"))


def evalute_model_on_different_countries(df: pd.DataFrame) -> None:
    # fit a model over the entire subsets of records from Israel using k=3
    israel_df = df[df["Country"] == "Israel"]
    israel_df = israel_df.drop(["Country", "City", "Date"], axis=1)
    X, y = israel_df.drop("Temp", axis=1), israel_df.Temp
    X = X["DayOfYear"].values
    model = PolynomialFitting(k=5)
    model.fit(X, y)

    # evaluate the model on different countries
    countries = df["Country"].unique()
    countries = countries[countries != "Israel"]
    losses = []
    for country in countries:
        country_df = df[df["Country"] == country]
        country_df = country_df.drop(["Country", "City", "Date"], axis=1)
        X_country, y_country = country_df.drop("Temp", axis=1), country_df.Temp
        X_country = X_country["DayOfYear"].values
        loss = model.loss(X_country, y_country)
        print(f"Loss for {country}: {loss}")
        losses.append(loss)

    # plot loss vs country using bar plot
    tools.config_plot(
        title="Loss vs Country", xlabel="Country", ylabel="Loss", figsize=(10, 6)
    )
    plt.bar(countries, losses, color="skyblue", edgecolor="black")
    plt.xticks(rotation=45)
    tools.save_and_done_plot(os.path.join(PLOTS_DIR, "loss_vs_country.png"))


if __name__ == "__main__":
    # Question 2 - Load and preprocessing of city temperature dataset
    df = load_data("city_temperature.csv")
    # create directory for plots
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.random.seed(130)
    # Question 3 - Exploring data for specific country
    explore_israel_temperature(df)
    # Question 4 - Exploring differences between countries
    explore_differences_between_countries(df)
    # Question 5 - Fitting model for different values of `k`
    fit_model_for_different_k(df)
    # Question 6 - Evaluating fitted model on different countries
    evalute_model_on_different_countries(df)

    pass
