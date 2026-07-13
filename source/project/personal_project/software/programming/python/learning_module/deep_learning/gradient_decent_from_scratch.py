import marimo

__generated_with = "0.19.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    from sklearn.datasets import make_regression

    return make_regression, np


@app.cell
def _(np):
    class GDRegressor:
        def __init__(self, learning_rate, epochs):
            self.m = 28.12597332
            self.b = -120
            self.lr = learning_rate
            self.epochs = epochs

        def fit(self, X, y):
            # Calculate b

            for i in range(self.epochs):
                loss_slope_b = -2 * np.sum(y - (self.m) * (X.ravel()) - self.b)
                self.b = self.b - (self.lr) * loss_slope_b

                print(f"m = {self.m}, and b = {self.b}")

        def predict(self, X):
            return self.m * X + self.b

    return (GDRegressor,)


@app.cell
def _(make_regression):
    X, y = make_regression(
        n_samples=100,
        n_features=1,
        n_informative=1,
        n_targets=1,
        noise=20,
        random_state=42,
    )
    return X, y


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, y_train


@app.cell
def _(GDRegressor, X_train, y_train):
    gd = GDRegressor(0.001, 100)
    gd.fit(X_train, y_train)
    return


@app.cell
def _(np):
    class GDRegressors:
        def __init__(self, learning_rate, epochs):
            self.m = 28.12597332
            self.b = -120
            self.lr = learning_rate
            self.epochs = epochs

        def fit(self, X, y):
            # Calculate b

            for i in range(self.epochs):
                loss_slope_b = -2 * np.sum(y - (self.m) * (X.ravel()) - self.b)
                self.b = self.b - (self.lr) * loss_slope_b

                loss_slope_m = -2 * np.sum(
                    (y - (self.m) * (X.ravel()) - self.b) * X.ravel()
                )
                self.m = self.m - (self.lr) * loss_slope_m

                print(f"m = {self.m}, and b = {self.b}")

        def predict(self, X):
            return self.m * X + self.b

    return (GDRegressors,)


@app.cell
def _(GDRegressors):
    gds = GDRegressors(0.001, 70)
    return (gds,)


@app.cell
def _(X_train, gds, y_train):
    gds.fit(X_train, y_train)
    return


if __name__ == "__main__":
    app.run()
