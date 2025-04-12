import os
import matplotlib.pyplot as plt
from typing import NoReturn
import numpy as np
import pandas as pd

# from linear_regression import LinearRegression

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
    

    X = X.drop("long", axis=1)
    X.loc[X["yr_renovated"] == 0, "yr_renovated"] = X["yr_built"]
    X["date"] = pd.to_datetime(X["date"]).astype("int64")/1e18
    # Loop through rows and assign new IDs to NaNs
    for i in X.index:
        if pd.isna(X.at[i, "id"]):
            new_id = X["id"].max() + 1
            X.at[i, "id"] = new_id

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
    # Combine X and y into one DataFrame for consistent row filtering
    data = pd.concat([X, y.rename("__y__")], axis=1)

    # Remove rows with NaN or Inf in any column
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    y = data["__y__"]
    X = data.drop(columns="__y__")
    
    # Keep only rows where date is non-negative
    valid_rows = X["date"] >= 0
    X = X[valid_rows]
    y = y[valid_rows]

    return X, y


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
    return X


def feature_evaluation(
    X: pd.DataFrame, y: pd.Series, output_path: str = "."
) -> NoReturn:
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
        std_x = x_feat.std(ddof=1)
        std_y = y.std(ddof=1)
        cov_xy = x_feat.cov(y)
        
        pearson_corr = cov_xy / (std_x * std_y) if std_x != 0 and std_y != 0 else 0
        plt.figure()
        plt.scatter(x_feat, y)
        # plt.ylim(0, 2_000_000)
        plt.title(f"{feature} vs Price\nPearson Correlation: {pearson_corr:.3f}")
        plt.xlabel(feature)
        plt.ylabel("Price")
        plt.savefig(f"{output_path}/{feature}_vs_price.png")
        plt.close()



def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
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

def question_6_improvement_tracker(
    mean_losses: list,
    std_losses: list,
    output_path: str = r"C:\Users\lotem\Documents\uni\courses\Y2S2\IML\EX1\track",
) -> NoReturn:

    csv_path = os.path.join(output_path, "mse_vs_training_size.csv")
    last_result = pd.DataFrame({
    "Percentage": [1.0],
    "Mean_Loss": [mean_losses[-1]],
    "Std_Loss": [std_losses[-1]]
    })

    # אם הקובץ קיים - נטען אותו, נצרף שורה חדשה ונשמור
    if os.path.exists(csv_path):
        prev_data = pd.read_csv(csv_path)
        new_data = pd.concat([prev_data, last_result], ignore_index=True)
        new_data.to_csv(csv_path, index=False)
    else:
        last_result.to_csv(csv_path, index=False)
    

def question_6_implementation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: str = ".",
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
            # Sample p% of the overall training data
            sample_size = round(len(X_train) * (p))  
            random_state = i + int(p * 100)
            X_sampled = X_train.sample(sample_size, random_state=random_state)
            y_sampled = y_train.loc[X_sampled.index]

            # Fit linear model (including intercept) over sampled set
            model = LinearRegression(include_intercept=True)
            model.fit(X_sampled.values, y_sampled.values)
            # print(model.coefs_)
            loss = model.loss(X_test.values, y_test.values)
            losses.append(loss)

        mean_losses.append(np.mean(losses))
        std_losses.append(np.std(losses))

    plt.figure()
    plt.plot(percentages, mean_losses, label="Mean Loss")
    plt.fill_between(
        percentages,
        np.array(mean_losses) - 2 * np.array(std_losses),
        np.array(mean_losses) + 2 * np.array(std_losses),
        alpha=0.2,
        label="Variance",
    )
    plt.xlabel("Training Size (%)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE vs Training Size")
    plt.legend()
    plt.savefig(f"{output_path}/mse_vs_training_size.png")
    plt.close()
    question_6_improvement_tracker(mean_losses, std_losses)

if __name__ == "__main__":
    csv_path = r"C:\Users\lotem\Documents\uni\courses\Y2S2\IML\EX1\additional_files\house_prices.csv"
    # csv_path = r"house_prices.csv"
    df = pd.read_csv(csv_path)
    X, y = df.drop("price", axis=1), df.price
    # set random seed
    np.random.seed(130)

    # Question 2 - split train test randomly to 80% train and 20% test
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)

    # Question 3 - preprocessing of housing prices train dataset
    X_train, y_train = preprocess_train(X_train, y_train)
    # go through every line in X_train and X_test and change floor to 3 if it's 4
    # and change floor to 2 if it's 1, now do this:
    # X_train.loc[X_train['floor'] == 4, 'floor'] = 3

    # for every line, if yr_renovated is 0, change it to yr_built of the same line
    # this works im not sure how:
    # X.loc[X["yr_renovated"] == 0, "yr_renovated"] = X["yr_built"]

    # Question 4 - Feature evaluation of train dataset with respect to response
    output_path = r"C:\Users\lotem\Documents\uni\courses\Y2S2\IML\EX1\plots"
    # output_path = "plots"
    feature_evaluation(X_train, y_train, output_path=output_path)

    # Question 5 - preprocess the test data
    X_test = preprocess_test(X_test)

    # Question 6 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    output_path_question6 = r"C:\Users\lotem\Documents\uni\courses\Y2S2\IML\EX1\question6plots"
    question_6_implementation(
        X_train, y_train, X_test, y_test, output_path=output_path_question6
    )

    # find all lines in X that id in not a number  greater than 10 (but dont check > 10 )


    




    
