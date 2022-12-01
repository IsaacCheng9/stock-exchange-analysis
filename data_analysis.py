"""
Data Set: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data?select=indexProcessed.csv

Ideas:
    - Find common measures such as volatility and momentum.
    - Use LASSO regression for stock market forecasting.
    - Use ridge regression for stock market forecasting.
    - Compare the results from LASSO and ridge regression - LASSO should be
      better according to papers?
"""
import pandas
from sklearn.model_selection import train_test_split


def main():
    stock_data = load_stock_data("data/index_processed.csv")
    train_test_per_index = split_train_test_per_index(stock_data)
    print(train_test_per_index.keys())


def load_stock_data(file_name: str) -> pandas.DataFrame:
    stock_data = pandas.read_csv(file_name)
    return stock_data


def split_train_test_per_index(
    data_set: pandas.DataFrame, training_proportion: float = 0.8
) -> dict:
    """
    Splits the data set into training and test data sets for each stock index.

    Args:
        data_set: The data set to split.
        training_proportion: The proportion of the data set to be used for the
                             training data set.

    Returns:
        A dictionary of the training and test data sets for each index.
    """
    # Split the data set by 'Index' column.
    data_frames_per_index = {
        key: data_set.loc[value]
        for key, value in data_set.groupby("Index").groups.items()
    }
    # Generate a training and test data set for each index.
    train_test_per_index = {}
    for key, value in data_frames_per_index.items():
        training_data, test_data = train_test_split(
            value, train_size=training_proportion
        )
        train_test_per_index[key] = [training_data, test_data]

    return train_test_per_index


def build_lasso_regression_model(training_data: pandas.DataFrame):
    """
    Builds a LASSO regression model to predict stock market prices.
    """
    pass


def build_ridge_regression_model(training_data: pandas.DataFrame):
    """
    Builds a ridge regression model to predict stock market prices.
    """
    pass


if __name__ == "__main__":
    main()
