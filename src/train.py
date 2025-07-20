import feast
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

mlflow.set_tracking_uri("http://localhost:5000")

# iris data
entity_df = pd.read_csv("data/iris_data_1.csv")
entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

# Connect to our feature store provider
fs = feast.FeatureStore(repo_path="iris_feast")

# Retrieve training data
training_df = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "iris_feature:sepal_length",
        "iris_feature:sepal_width",
        "iris_feature:petal_length",
        "iris_feature:petal_width",
    ],
).to_df()

print("Data Retrieved")
print("----- Feature schema -----\n")
print(training_df.info())

print()
print("----- Example features -----\n")
print(training_df.head(5))

# Merge target column (species)
training_df["species"] = entity_df["species"]

# Features and target
X = training_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = training_df["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    # Define and train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Log model parameters
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("test_size", 0.2)

    # Log metrics
    mlflow.log_metric("accuracy", acc)

    # Save and log model
    dump(model, "iris_model.joblib")
    mlflow.sklearn.log_model(model, "model")
    print("\nModel saved and logged to MLflow as iris_model.joblib")
