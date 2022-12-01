"""
Data Set: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data?select=indexProcessed.csv

Ideas:
    - Find common measures such as volatility and momentum.
    - Use LASSO regression for stock market forecasting.
    - Use ridge regression for stock market forecasting.
    - Compare the results from LASSO and ridge regression - LASSO should be
      better according to papers?
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    stock_data = load_stock_data("data/index_processed.csv")
    training_data, test_data = split_data_set(stock_data)
    print(training_data)
    print(test_data)


def load_stock_data(file_name: str) -> pd.DataFrame:
    stock_data = pd.read_csv(file_name)
    return stock_data


def split_data_set(data_set: pd.DataFrame, training_proportion: float = 0.8) -> list:
    """
    Splits the data set into training and test data sets.

    Args:
        data_set: The data set to split.
        training_proportion: The proportion of the data set to be used for the
                             training data set.

    Returns:
        A list of the training and test data sets.
    """
    training_data, test_data = train_test_split(
        data_set, train_size=training_proportion
    )
    return [training_data, test_data]


if __name__ == "__main__":
    main()
