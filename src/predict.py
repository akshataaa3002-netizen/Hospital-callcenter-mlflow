import mlflow
import pandas as pd

from preprocess import load_and_preprocess_data


def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        target_encoder,
        feature_encoders,
    ) = load_and_preprocess_data(
        "Data/synthetic_patient_call_routing_dataset.csv"
    )

    # Load registered model from MLflow Model Registry
    model = mlflow.pyfunc.load_model(
        "models:/Hospital_Call_Center_Routing_Model/1"
    )

    # Select one sample
    sample = X_test.iloc[[0]]

    print(sample)
    print(sample.dtypes)

    # Predict
    prediction = model.predict(sample)

    prediction = prediction.astype(int)

    predicted_department = target_encoder.inverse_transform(prediction)[0]

    print("\n" + "=" * 50)
    print("PREDICTION USING REGISTERED MODEL")
    print("=" * 50)

    print("\nInput Features:")
    print(sample)

    print("\nPredicted Department:")
    print(predicted_department)

    print("=" * 50)


if __name__ == "__main__":
    main()
