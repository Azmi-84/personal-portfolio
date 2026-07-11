import marimo

__generated_with = "0.19.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_california_housing
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    return (
        GradientBoostingRegressor,
        fetch_california_housing,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        train_test_split,
    )


@app.cell
def _(fetch_california_housing):
    df = fetch_california_housing(as_frame=True)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    X = df.data
    X
    return (X,)


@app.cell
def _(df):
    y = df.target
    y
    return (y,)


@app.cell
def _(X, train_test_split, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(GradientBoostingRegressor):
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    return (gbr,)


@app.cell
def _(X_train, gbr, y_train):
    gbr.fit(X_train, y_train)
    return


@app.cell
def _(X_test, gbr):
    y_test_predict = gbr.predict(X_test)
    return (y_test_predict,)


@app.cell
def _(mean_squared_error, r2_score, y_test, y_test_predict):
    mse = mean_squared_error(y_test, y_test_predict)
    r2 = r2_score(y_test, y_test_predict)

    print(f"Mean Squared Error: {mse}, and R2 Score: {r2}")
    return


@app.cell
def _(np, plt, y_test, y_test_predict):
    # Calculate the absolute error for coloring
    error = np.abs(y_test - y_test_predict)

    plt.figure(figsize=(10, 6))

    # Use a colormap (c=error) to show which points are most inaccurate
    scatter = plt.scatter(y_test, y_test_predict, c=error, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Absolute Error')

    # The "Perfect Prediction" line
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="-.", color="red", label="Perfect Fit")

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted (Colored by Error)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, y_test, y_test_predict):
    plt.figure(figsize=(10, 6))

    plt.plot(range(len(y_test)), y_test, label="Actual", color="blue", marker='o', linestyle='')

    plt.plot(range(len(y_test_predict)), y_test_predict, label="Predicted", color="orange", marker='x', linestyle='')

    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted Comparison")
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(GradientBoostingRegressor, X_test, X_train, r2_score, y_test, y_train):
    # Effect of learning_rate

    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.8, 1]
    learning_rate_results = []

    for lr in learning_rates:
        _model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=lr,
            max_depth=3,
            random_state=42
        )
        _model.fit(X_train, y_train)
        _y_test_predicts = _model.predict(X_test)
        learning_rate_results.append((lr, r2_score(y_test, _y_test_predicts)))
    return (learning_rate_results,)


@app.cell
def _(learning_rate_results, pd):
    learning_rate_results_df = pd.DataFrame(learning_rate_results, columns=["Learning Rate", "R2 Score"])
    learning_rate_results_df
    return (learning_rate_results_df,)


@app.cell
def _(learning_rate_results_df, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rate_results_df['Learning Rate'], learning_rate_results_df['R2 Score'], marker='o', color='red')

    plt.xlabel('Learning Rate')
    plt.ylabel('R2 Score')
    plt.title('Learning Rate vs R2 Score')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(GradientBoostingRegressor, X_test, X_train, r2_score, y_test, y_train):
    # Effect of max_depth

    max_depths = [1, 2, 3, 5, 7, 9, 11, 13]
    max_depths_results = []

    for md in max_depths:
        _model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.2,
            max_depth=md,
            random_state=42
        )
        _model.fit(X_train, y_train)
        _y_test_predicts = _model.predict(X_test)
        max_depths_results.append((md, r2_score(y_test, _y_test_predicts)))
    return (max_depths_results,)


@app.cell
def _(max_depths_results, pd):
    max_depths_results_df = pd.DataFrame(max_depths_results, columns=["Max Depth", "R2 Score"])
    max_depths_results_df
    return (max_depths_results_df,)


@app.cell
def _(max_depths_results_df, plt):
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths_results_df['Max Depth'], max_depths_results_df['R2 Score'], marker='o', color='red')

    plt.xlabel('Max Depth')
    plt.ylabel('R2 Score')
    plt.title('Max Depth vs R2 Score')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
