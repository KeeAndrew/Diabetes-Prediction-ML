# diabetes_svm.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main(csv_path: Path = Path("diabetes.csv")):
    if not csv_path.exists():
        print(f"[ERROR] Could not find {csv_path.resolve()}")
        print("Place diabetes.csv in this folder or pass a path: python diabetes_svm.py path/to/diabetes.csv")
        sys.exit(1)

    # Load data
    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required_cols = [
        "Pregnancies","Glucose","BloodPressure","SkinThickness",
        "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing expected columns: {missing}")
        sys.exit(1)

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=2
    )

    # Build pipeline: scale -> linear SVM
    clf = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="linear", probability=False, random_state=2))
    ])

    # Train
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_test)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test  accuracy: {test_acc:.3f}")
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred_test, digits=3))
    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, y_pred_test))

    # Example single prediction (your sample)
    example = np.array([[5, 166, 72, 19, 175, 25.8, 0.587, 51]])
    example_pred = clf.predict(example)[0]
    print("\nSample prediction for (5,166,72,19,175,25.8,0.587,51):",
          "Diabetic" if example_pred == 1 else "Not diabetic")

if __name__ == "__main__":
    # Allow an optional path arg: python diabetes_svm.py path/to/diabetes.csv
    path_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("diabetes.csv")
    main(path_arg)
