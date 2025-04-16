
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def config_plot(title: str, xlabel: str, ylabel: str, figsize= (10,5)) -> None:
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    

def save_and_done_plot(filename: str) -> None:
    """
    Save plot to file and close it
    Parameters
    ----------
    filename : str
        Path to save the plot to
    """
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float):
    """
    Split dataset into train and test set
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    test_size : float (default 0.2)
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
    # Randomly shuffle the data
    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X.iloc[shuffled_indices]
    y_shuffled = y.iloc[shuffled_indices]
    # Calculate the split index
    split_index = int(len(X) * (1 - test_size))
    # Split the data into train and test sets
    X_train = X_shuffled.iloc[:split_index]
    y_train = y_shuffled.iloc[:split_index]
    X_test = X_shuffled.iloc[split_index:]
    y_test = y_shuffled.iloc[split_index:]
    return X_train, y_train, X_test, y_test