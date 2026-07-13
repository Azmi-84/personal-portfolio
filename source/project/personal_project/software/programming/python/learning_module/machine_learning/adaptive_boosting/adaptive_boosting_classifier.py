import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    return (
        AdaBoostClassifier,
        DecisionTreeClassifier,
        accuracy_score,
        classification_report,
        load_breast_cancer,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(load_breast_cancer):
    X, y = load_breast_cancer(return_X_y=True)
    print(X.shape, y.shape)
    return X, y


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    dtc_model = DecisionTreeClassifier(max_depth=1, random_state=42)
    dtc_model.fit(X_train, y_train)
    return (dtc_model,)


@app.cell
def _(X_test, dtc_model):
    y_test_pred_dtc = dtc_model.predict(X_test)
    return (y_test_pred_dtc,)


@app.cell
def _(accuracy_score, y_test, y_test_pred_dtc):
    accuracy_score(y_test, y_test_pred_dtc)
    return


@app.cell
def _(AdaBoostClassifier, DecisionTreeClassifier):
    # AdaBoost Model
    base_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    adaboost_clf = AdaBoostClassifier(estimator=base_learner, n_estimators=1000, random_state=42)
    return adaboost_clf, base_learner


@app.cell
def _(X_train, adaboost_clf, y_train):
    adaboost_clf.fit(X_train, y_train)
    return


@app.cell
def _(X_test, adaboost_clf):
    y_test_pred_adaboost = adaboost_clf.predict(X_test)
    return (y_test_pred_adaboost,)


@app.cell
def _(accuracy_score, y_test, y_test_pred_adaboost):
    accuracy_score(y_test, y_test_pred_adaboost)
    return


@app.cell
def _(classification_report, y_test, y_test_pred_adaboost):
    print(classification_report(y_test, y_test_pred_adaboost))
    return


@app.cell
def _():
    estimators = [_i for _i in range(10, 210, 10)]
    estimators
    return (estimators,)


@app.cell
def _(
    AdaBoostClassifier,
    X_test,
    X_train,
    accuracy_score,
    base_learner,
    estimators,
    y_test,
    y_train,
):
    estimator_acc = []

    for _i in estimators:
        _model = AdaBoostClassifier(
            estimator=base_learner,
            n_estimators=_i,
            learning_rate=0.3,
            random_state=42
        )

        _model.fit(X_train, y_train)
        estimator_acc.append(accuracy_score(y_test, _model.predict(X_test)))
    return (estimator_acc,)


@app.cell
def _(estimator_acc, estimators, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(estimators, estimator_acc, marker='o')
    plt.xlabel("Numerical Estimators")
    plt.ylabel("Accuracy")
    plt.title("Effect of n_estimators")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(np):
    learning_rate = [float(_i) for _i in np.arange(0.1, 2.1, 0.1).round(2)]
    learning_rate
    return (learning_rate,)


@app.cell
def _(
    AdaBoostClassifier,
    X_test,
    X_train,
    accuracy_score,
    base_learner,
    learning_rate,
    y_test,
    y_train,
):
    learning_rate_acc = []

    for _i in learning_rate:
        _model = AdaBoostClassifier(
            estimator=base_learner,
            n_estimators=100,
            learning_rate=_i,
            random_state=42
        )

        _model.fit(X_train, y_train)
        learning_rate_acc.append(accuracy_score(y_test, _model.predict(X_test)))
    return (learning_rate_acc,)


@app.cell
def _(learning_rate, learning_rate_acc, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rate, learning_rate_acc, marker='o')
    plt.xlabel("Learning Rate")
    plt.ylabel("Accuracy")
    plt.title("Effect of learning_rate")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(AdaBoostClassifier, base_learner, estimators, learning_rate):
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        'n_estimators': estimators,
        'learning_rate': learning_rate,
        'estimator__max_depth': [1, 2]
    }

    grid = GridSearchCV(
        estimator=AdaBoostClassifier(
            estimator=base_learner,
            random_state=42
        ),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1
    )
    return (grid,)


@app.cell
def _(X_train, grid, y_train):
    grid.fit(X_train, y_train)
    return


@app.cell
def _(X_test, grid):
    best_model = grid.best_estimator_
    y_test_pred_grid = best_model.predict(X_test)
    return (y_test_pred_grid,)


@app.cell
def _(grid):
    print("Best parameters found: ")
    print(grid.best_params_)
    return


@app.cell
def _(accuracy_score, y_test, y_test_pred_grid):
    print("Test accuracy: ", accuracy_score(y_test, y_test_pred_grid))
    return


@app.cell
def _(classification_report, y_test, y_test_pred_grid):
    print(classification_report(y_test, y_test_pred_grid))
    return


if __name__ == "__main__":
    app.run()
