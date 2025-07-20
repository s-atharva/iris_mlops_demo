import pandas as pd


def test_column_count():
    df = pd.read_csv('iris_data.csv')
    assert df.shape[1] == 6, f"Expected 6 columns, but found {df.shape[1]}"
