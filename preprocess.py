import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data(filepath):

    print(">>> preprocess.py is running")

    df = pd.read_csv(filepath)

    df.drop(columns=["Call_ID", "Call_Timestamp"], inplace=True)

    X = df.drop(columns=["Recommended_Department"])
    y = df["Recommended_Department"]

    feature_encoders = {}

    for col in X.columns:
        if X[col].dtype == "object":
            encoder = LabelEncoder()
            X[col] = encoder.fit_transform(X[col].astype(str))
            feature_encoders[col] = encoder

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    result = (
        X_train,
        X_test,
        y_train,
        y_test,
        target_encoder,
        feature_encoders,
    )

    print(">>> Returning:", len(result), "values")

    return result