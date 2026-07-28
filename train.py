import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)

from preprocess import load_and_preprocess_data


def main():

    # Load and preprocess data
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

    # Create MLflow experiment
    mlflow.set_experiment("Hospital_Call_Center_Routing")

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # Train every model
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Metrics
            accuracy = accuracy_score(y_test, y_pred)

            precision = precision_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            recall = recall_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average="weighted",
                zero_division=0,
            )

            # Log parameters
            mlflow.log_param("Algorithm", model_name)
            mlflow.log_param("Model_Class", type(model).__name__)

            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)

            if hasattr(model, "random_state"):
                mlflow.log_param("random_state", model.random_state)

            # Log metrics
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))

            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=target_encoder.classes_,
            ).plot(ax=ax, xticks_rotation=45)

            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close(fig)

            mlflow.log_artifact("confusion_matrix.png")

            # Classification Report
            report = classification_report(
                y_test,
                y_pred,
                target_names=target_encoder.classes_,
                zero_division=0,
            )

            with open("classification_report.txt", "w") as file:
                file.write(report)

            mlflow.log_artifact("classification_report.txt")

            # Save model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=model_name.lower().replace(" ", "_"),
            )

            # Console Output
            print("\n" + "=" * 60)
            print(f"MODEL : {model_name}")
            print("=" * 60)
            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print("=" * 60)


if __name__ == "__main__":
    main()