import marimo

__generated_with = "0.19.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    from xgboost import XGBClassifier
    return (
        XGBClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        load_breast_cancer,
        pd,
        plt,
        train_test_split,
    )


@app.cell
def _(load_breast_cancer):
    data = load_breast_cancer()
    return (data,)


@app.cell
def _(data, pd):
    X = pd.DataFrame(data.data, columns=data.feature_names)
    X
    return (X,)


@app.cell
def _(data, pd):
    y = pd.Series(data.target)
    y
    return (y,)


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(XGBClassifier):
    xgb_model = XGBClassifier(
        n_estimators = 100,
        max_depth = 3,
        learning_rate = 0.1,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = "binary:logistic",
        eval_metric = "logloss",
        random_state = 42
    )
    return (xgb_model,)


@app.cell
def _(X_train, xgb_model, y_train):
    xgb_model.fit(X_train, y_train)
    return


@app.cell
def _(X_test, xgb_model):
    y_test_pred = xgb_model.predict(X_test)
    return (y_test_pred,)


@app.cell
def _(accuracy_score, y_test, y_test_pred):
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")
    return


@app.cell
def _(confusion_matrix, y_test, y_test_pred):
    print(confusion_matrix(y_test, y_test_pred))
    return


@app.cell
def _(classification_report, y_test, y_test_pred):
    print(classification_report(y_test, y_test_pred))
    return


@app.cell
def _(X_test, X_train, y_test, y_train):
    # Early Stopping (Overfitting Control)

    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
        "eta": 0.05, # learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42
    }

    model_xgb = xgb.train(
        params = params,
        dtrain = dtrain,
        num_boost_round = 500,
        evals = [(dtest, "eval")],
        early_stopping_rounds = 50
    )
    return dtest, model_xgb


@app.cell
def _(dtest, model_xgb):
    y_test_prob_xgb = model_xgb.predict(dtest)
    y_test_pred_xgb = (y_test_prob_xgb >= 0.5).astype(int)
    return (y_test_pred_xgb,)


@app.cell
def _(accuracy_score, y_test, y_test_pred_xgb):
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred_xgb)}")
    return


@app.cell
def _(classification_report, y_test, y_test_pred_xgb):
    print(classification_report(y_test, y_test_pred_xgb))
    return


@app.cell
def _(confusion_matrix, y_test, y_test_pred_xgb):
    print(confusion_matrix(y_test, y_test_pred_xgb))
    return


@app.cell
def _(model_xgb):
    print("Early Stopping Information")
    print(f"Best iteration: {model_xgb.best_iteration}")
    print(f"Best score (logloss): {model_xgb.best_score}")
    return


@app.cell
def _(X, plt, xgb_model):
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, xgb_model.feature_importances_)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
