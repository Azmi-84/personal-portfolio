# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib==3.10.8",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "scikit-learn==1.8.0",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    from sklearn.datasets import load_breast_cancer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    return (
        LogisticRegression,
        StandardScaler,
        accuracy_score,
        classification_report,
        confusion_matrix,
        load_breast_cancer,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(np):
    hours_studied = np.array([_i for _i in range(1, 11, 1)]).reshape(-1, 1)
    pass_exam = np.array([0 if _i <= 5 else 1 for _i in range(1, 11, 1)])
    return hours_studied, pass_exam


@app.cell
def _(hours_studied, pass_exam, pd):
    toy_df = pd.DataFrame({
        'hourse_studied': hours_studied.flatten(),
        'pass_exam': pass_exam
    })

    toy_df
    return


@app.cell
def _(LogisticRegression, hours_studied, pass_exam):
    toy_model = LogisticRegression()
    toy_model.fit(hours_studied, pass_exam)
    return (toy_model,)


@app.cell
def _(toy_model):
    print("Intercept (b): ", toy_model.intercept_)
    print("Cofficient (w): ", toy_model.coef_)
    return


@app.cell
def _(np, toy_model):
    # Predict probabilities for a range of study hours

    hours_grid = np.linspace(0, 10, 200).reshape(-1, 1)
    pass_prob = toy_model.predict_proba(hours_grid)[:, 1]
    return hours_grid, pass_prob


@app.cell
def _(hours_grid, hours_studied, pass_exam, pass_prob, plt):
    # Plot data points and probability curve

    plt.figure(figsize=(8, 6))
    plt.scatter(hours_studied, pass_exam, label="Actual data (0 or 1)")
    plt.plot(hours_grid, pass_prob, label="Predicted probability of passing")
    plt.xlabel("Hours Studied")
    plt.ylabel("Probability of Passing")
    plt.title("Logistic Regression on Toy Data")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(np, toy_model):
    # Predict for a few example students

    example_hours = np.array([[_i] for _i in range(1, 10) if _i % 2 == 0])
    example_probs = toy_model.predict_proba(example_hours)[:, 1 ]
    example_predict = toy_model.predict(example_hours)
    return example_hours, example_predict, example_probs


@app.cell
def _(example_hours, example_predict, example_probs, np, pd):
    results = pd.DataFrame({
        "hours_studied": example_hours.flatten(),
        "predicted_probability_pass": np.round(example_probs, 4),
        "predicted_classs": example_predict
    })
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(np):
    # Sigmodi Function

    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    return (sigmoid,)


@app.cell
def _(np, pd, sigmoid):
    # Check some values

    z_values = np.linspace(-30, 30, 50)
    sig_values = sigmoid(z_values)

    sig_df = pd.DataFrame({
        "z": z_values,
        "sigmoid_z": np.round(sig_values, 4)
    })

    sig_df
    return sig_values, z_values


@app.cell
def _(plt, sig_values, z_values):
    plt.scatter(z_values, sig_values)
    plt.xlabel("Z Values")
    plt.ylabel("Sigmoid Values")
    plt.title("Sigmoid Function")
    plt.axhline(0.5, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(load_breast_cancer):
    # Logistic Regression application on real life dataset

    # Load dataset
    data = load_breast_cancer()
    X_full = data.data
    y_full = data.target

    print("Feature matrix shape: ", X_full.shape)
    print("Target shape: ", y_full.shape)
    print("Classes: ", data.target_names)
    return X_full, data, y_full


@app.cell
def _(X_full, data, pd, y_full):
    feature_names = data.feature_names

    df = pd.DataFrame(X_full, columns=feature_names)
    df["target"] = y_full
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.describe().T
    return


@app.cell
def _(df):
    # Target class count

    df["target"].value_counts()
    return


@app.cell
def _(X_full, train_test_split, y_full):
    # Train test split

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, train_size=0.8, random_state=42, stratify=y_full
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(StandardScaler, X_test, X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled, X_train_scaled


@app.cell
def _(LogisticRegression, X_train_scaled, y_train):
    # Train logistic regression on the scaled data

    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train_scaled, y_train)
    return (clf,)


@app.cell
def _(X_test_scaled, accuracy_score, clf, y_test):
    y_test_pred = clf.predict(X_test_scaled)
    y_test_proba = clf.predict_proba(X_test_scaled)[:, 1]

    print("Test Accuracy: ", accuracy_score(y_test, y_test_pred))
    return (y_test_pred,)


@app.cell
def _(confusion_matrix, y_test, y_test_pred):
    # Confusion matrix and classification report

    cm = confusion_matrix(y_test, y_test_pred)

    print("Confusion matrix:\n", cm)
    return


@app.cell
def _(classification_report, y_test, y_test_pred):
    print("\nClassification report:\n", classification_report(y_test, y_test_pred))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
