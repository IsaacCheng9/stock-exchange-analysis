"""
Data Set: https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data?select=indexProcessed.csv

Ideas:
    - Find common measures such as volatility and momentum.
    - Use LASSO regression for stock market forecasting.
    - Use ridge regression for stock market forecasting.
    - Compare the results from LASSO and ridge regression - LASSO should be
      better according to papers?
"""
from typing import List

import pandas as pd


def main():
    stock_data = load_stock_data("data/index_processed.csv")
    training_data, test_data = split_data_set(stock_data)
    print(training_data)
    print(test_data)


def load_stock_data(file_name: str) -> pd.DataFrame:
    stock_data = pd.read_csv(file_name)
    return stock_data


def split_data_set(
    stock_data: pd.DataFrame, training_proportion: float = 0.8
) -> List[pd.DataFrame]:
    split_index = int(len(stock_data) * training_proportion)
    # TODO: Improve the stock data split, as it's currently not random.
    training_data = stock_data[:split_index]
    test_data = stock_data[split_index:]
    return [training_data, test_data]


if __name__ == "__main__":
    main()
