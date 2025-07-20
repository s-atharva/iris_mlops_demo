from sklearn.datasets import load_iris
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from datetime import datetime, timedelta

# Load Iris data
iris = load_iris(as_frame=True)
df = iris.frame

# Add event_timestamp column as real timestamp (not string)
df["event_timestamp"] = pd.Timestamp.now() - pd.to_timedelta(range(len(df)), unit="d")

# Add flower_id using OrdinalEncoder on target column
encoder = OrdinalEncoder()
df["flower_id"] = encoder.fit_transform(df[["target"]]).astype(int)

# Rename and select columns
df = df.rename(columns={
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
    "target": "species"
})
# df = df[["event_timestamp", "flower_id", "species"]]
df = df[["event_timestamp", "flower_id", "sepal_length", "sepal_width", "petal_length", "petal_width"]]

# Save to CSV (event_timestamp will retain datetime format)
df.to_csv("iris_data.csv", index=False)
