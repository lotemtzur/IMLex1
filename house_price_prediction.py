import os
import matplotlib.pyplot as plt
from typing import NoReturn
import numpy as np
import pandas as pd

from linear_regression import LinearRegression
import tools


def preprocess_columns(X: pd.DataFrame):
    """
    preprocess columns of the data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A clean, preprocessed version of the data
    """

    X["date"] = pd.to_datetime(X["date"])
    X["sqft_living_squared"] = X["sqft_living"] ** 2
    X["sqft_above_squared"] = X["sqft_above"] ** 2
    X["days_since_first_house"] = (X["date"] - X["date"].min()).dt.days
    X["grade_squared"] = X["grade"] ** 2
    X = X.drop(columns=["id", "date"])

    return X


def preprocess_train(X: pd.DataFrame, y: pd.Series):
    """
    preprocess training data.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data
    y: pd.Series

    Returns
    -------
    A clean, preprocessed version of the data
    """

    X = preprocess_columns(X)

    # Remove rows with NaN or Inf in X or y
    mask_valid_values = ~(
        X.replace([np.inf, -np.inf], np.nan).isna().any(axis=1) | y.isna()
    )

    # Keep valid renovated vs built
    mask_renovation = (X["yr_renovated"] == 0) | (X["yr_built"] < X["yr_renovated"])

    # Ensure key features are non-negative
    mask_non_negative = (
        X[["bedrooms", "sqft_living", "bathrooms", "floors"]] >= 0
    ).all(axis=1)

    # Combine all masks
    final_mask = mask_valid_values & mask_renovation & mask_non_negative

    return X[final_mask], y[final_mask]


def preprocess_test(X: pd.DataFrame):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    Parameters
    ----------
    X: pd.DataFrame
        the loaded data

    Returns
    -------
    A preprocessed version of the test data that matches the coefficients format.
    """
    X = preprocess_columns(X)
    # since we dont remove lines, we'll just fill the missing values with the mean of the column
    X = X.fillna(X.mean(numeric_only=True))
    return X


def preprocess_test_y(y: pd.Series):
    """
    preprocess test data. You are not allowed to remove rows from X, but only edit its columns.
    set missing values to the mean of the column.
    Parameters
    ----------
    y: pd.Series
        the loaded data

    Returns
    -------
    y : pd.Series
        with no missing values
    """
    y = y.fillna(y.mean())
    return y


def calc_pearson_corr(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """
    Calculate the Pearson correlation coefficient between each feature in X and the response y.
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    Returns
    -------
    pearson_corr : Series of shape (n_features,)
        Pearson correlation coefficients between each feature and the response
    """
    std_x = X.std(ddof=1)
    std_y = y.std(ddof=1)
    cov_xy = X.cov(y)

    return cov_xy / (std_x * std_y) if std_x != 0 and std_y != 0 else 0


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str) -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    This function will compute the Pearson Correlation between each of the features and the response:
        Pearson Correlation: ρ := (COV(X,Y))/(σX*σY) for X being one of the features, and Y the response
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    os.makedirs(output_path, exist_ok=True)

    for feature in X.columns:
        x_feat = X[feature]
        pearson_corr = calc_pearson_corr(x_feat, y)
        tools.config_plot(
            title=f"{feature} vs Price\nPearson Correlation: {pearson_corr:.3f}",
            xlabel=feature,
            ylabel="Price",
        )
        plt.scatter(x_feat, y)
        tools.save_and_done_plot(filename=f"{output_path}/{feature}_vs_price.png")


def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    """
    Split dataset into train and test set
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    test_size : float
        Proportion of the dataset to include in the test split

    Returns
    -------
    X_train : DataFrame of shape (n_train_samples, n_features)
        Training data

    y_train : array-like of shape (n_train_samples,)
        Training response vector

    X_test : DataFrame of shape (n_test_samples, n_features)
        Test data

    y_test : array-like of shape (n_test_samples,)
        Test response vector
    """
    return tools.train_test_split(X, y, test_size=test_size)


def fit_model_and_plot_loss(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str,
) -> NoReturn:
    """
    Fit model over increasing percentages of the overall training data
        For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
            1) Sample p% of the overall training data
            2) Fit linear model (including intercept) over sampled set
            3) Test fitted model over test set
            4) Store average and variance of loss over test set
        Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    Parameters
    ----------
    X_train : DataFrame of shape (n_samples, n_features)
        Training data

    y_train : array-like of shape (n_samples,)
        Training response vector

    X_test : DataFrame of shape (n_test_samples, n_features)
        Test data

    y_test : array-like of shape (n_test_samples,)
        Test response vector

    output_path : str (default ".")
        Path to folder in which plots are saved
    """

    os.makedirs(output_path, exist_ok=True)
    percentages = np.arange(0.1, 1.01, 0.01)
    mean_losses = []
    std_losses = []

    for p in percentages:
        losses = []
        for i in range(10):
            sample_size = round(len(X_train) * (p))
            random_state = i + int(p * 100)
            X_sampled = X_train.sample(sample_size, random_state=random_state)
            y_sampled = y_train.loc[X_sampled.index]

            model = LinearRegression(include_intercept=True)
            model.fit(X_sampled.values, y_sampled.values)
            loss = model.loss(X_test.values, y_test.values)
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    print(f"loss: {mean_losses[-1]:.3f}")
    tools.config_plot(
        title="MSE vs Training Size",
        xlabel="Training Size (%)",
        ylabel="Mean Squared Error (MSE)",
    )
    plt.plot(percentages, mean_losses, label="Mean Loss")
    plt.fill_between(
        percentages,
        np.array(mean_losses) - 2 * np.array(std_losses),
        np.array(mean_losses) + 2 * np.array(std_losses),
        alpha=0.2,
        label="Mean Loss ± 2 Std",
    )
    plt.legend()
    tools.save_and_done_plot(filename=f"{output_path}/loss_vs_training_size.png")


if __name__ == "__main__":
    csv_path = "house_prices.csv"
    plots_path = "house_prices_plots"
    df = pd.read_csv(csv_path)
    X, y = df.drop("price", axis=1), df.price
    # set random seed
    np.random.seed(43)

    # Question 2 - split train test randomly to 75% train and 25% test
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.25)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)

    # Question 4 - Feature evaluation of train dataset with respect to response
    feature_evaluation(X_train, y_train, output_path=plots_path)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # handle with null prices in test set
    y_test = preprocess_test_y(y_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    fit_model_and_plot_loss(X_train, y_train, X_test, y_test, output_path=plots_path)
