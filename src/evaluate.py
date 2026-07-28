import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

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

    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("=" * 50)

    print("\nClassification Report\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_encoder.classes_,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=target_encoder.classes_,
    ).plot(ax=ax, xticks_rotation=45)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()
