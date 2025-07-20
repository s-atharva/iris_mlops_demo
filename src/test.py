from feast import FeatureStore
import pandas as pd
import joblib
import mlflow

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")

with mlflow.start_run(run_name="Predict_Iris"):
    # Load Feature Store
    store = FeatureStore(repo_path="iris_feast")

    # Entity for prediction
    entity_df = pd.DataFrame.from_dict({
        "flower_id": [0],
        "event_timestamp": [pd.Timestamp.now()]
    })

    # Get online features
    features = store.get_online_features(
        features=[
            "iris_feature:sepal_length",
            "iris_feature:sepal_width",
            "iris_feature:petal_length",
            "iris_feature:petal_width"
        ],
        entity_rows=entity_df.to_dict(orient="records")
    ).to_df()

    print(features)

    # Prepare features
    X_pred = features.drop(columns=["flower_id"], errors="ignore")

    # Load model
    model = joblib.load("iris_model.joblib")
    expected_features = model.feature_names_in_
    X_pred = X_pred[expected_features]

    # Predict
    y_pred = model.predict(X_pred)

    print("Prediction:", y_pred)

    # Log prediction input and result
    mlflow.log_dict(X_pred.to_dict(orient="records")[0], "input_features.json")
    mlflow.log_param("predicted_class", str(y_pred[0]))
