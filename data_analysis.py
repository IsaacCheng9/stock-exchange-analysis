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


def main():
    print(load_stock_data("data/index_processed.csv"))


def load_stock_data(file_name: str) -> pd.DataFrame:
    stock_data = pd.read_csv(file_name)
    return stock_data.head()


def get_stock_prices():
    pass


if __name__ == "__main__":
    main()
