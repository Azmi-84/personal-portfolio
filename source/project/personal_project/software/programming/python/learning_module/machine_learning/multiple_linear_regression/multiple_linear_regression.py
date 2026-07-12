# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair==6.0.0",
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "pyarrow==22.0.0",
#     "scikit-learn==1.7.2",
#     "seaborn==0.13.2",
#     "vegafusion==2.0.3",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import fetch_california_housing
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Configure matplotlib for slightly nicer default plots
    plt.rcParams['figure.figsize'] = (8, 5)
    plt.rcParams['axes.grid'] = True


@app.cell
def _():
    california = fetch_california_housing(as_frame=True)

    # california.frame is a pandas DataFrame that already includes both the features and target column
    df = california.frame.copy()
    df
    return (df,)


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.describe().T
    return


@app.cell
def _():
    target_column = 'MedHouseVal'
    feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    return feature_columns, target_column


@app.cell
def _(df, feature_columns, target_column):
    X = df[feature_columns]
    y = df[target_column]
    return X, y


@app.cell
def _(df, feature_columns, target_column):
    corr_matrix = df[feature_columns + [target_column]].corr()
    corr_matrix
    return (corr_matrix,)


@app.cell
def _(corr_matrix):
    sns.heatmap(corr_matrix, vmin=-1, vmax=1)
    plt.suptitle("Correlation between Features and Target Feature")
    return


@app.cell
def _(X, y):
    # Train test split

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    return X_test, X_train, y_test, y_train


@app.cell
def _():
    linear_regression_model = LinearRegression()
    return (linear_regression_model,)


@app.cell
def _(X_train, linear_regression_model, y_train):
    linear_regression_model.fit(X_train, y_train)
    return


@app.cell
def _(feature_columns, linear_regression_model):
    print("Intercept (bias term): ", linear_regression_model.intercept_)
    print("\nCoefficients: ")

    for feature_name, coef in zip(feature_columns, linear_regression_model.coef_):
        print(f"{feature_name}: {coef}")
    return


@app.cell
def _(X_test, X_train, linear_regression_model, y_test):
    y_train_prediction = linear_regression_model.predict(X_train)
    y_test_prediction = linear_regression_model.predict(X_test)

    print(y_test_prediction[:5])
    print(y_test[:5])
    return y_test_prediction, y_train_prediction


@app.function
def regression_metrics(y_true, y_pred, label='Model'):
    mae = mean_absolute_error(y_true, y_pred)
    msr = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(msr)
    r2 = r2_score(y_true, y_pred)

    print(f'{label}\n')
    print('MAE: ', mae)
    print('RMSE: ', rmse)
    print('R2', r2)


@app.cell
def _(y_train, y_train_prediction):
    regression_metrics(y_train, y_train_prediction, label='LinearRegression (Train)')
    return


@app.cell
def _(y_test, y_test_prediction):
    regression_metrics(y_test, y_test_prediction, label='LinearRegression (Test)')
    return


@app.cell
def _(y_test, y_test_prediction):
    plt.figure()
    plt.scatter(y_test, y_test_prediction, alpha=0.3)
    plt.xlabel('Actual MedHouseVal')
    plt.ylabel('Predicted MedHouseVal')
    plt.title('Actual vs Predicted House Value (Test Set)')

    # Diagonal reference line
    min_val = min(y_test.min(), y_test_prediction.min())
    max_val = max(y_test.max(), y_test_prediction.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(y_test, y_test_prediction):
    residuals = y_test - y_test_prediction

    plt.figure()
    plt.scatter(y_test_prediction, residuals, alpha=0.3)
    plt.axhline(0, linestyle='--')
    plt.xlabel('Predicted MedHouseVal')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Plot (Test Set)')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, target_column):
    # Prepare a single feature for illustration: MedInc vs MedHouseVal

    X_single = df[['MedInc']]
    y_single = df[target_column]

    X_single_train, X_single_test, y_single_train, y_single_test = train_test_split(X_single, y_single, train_size=0.8, random_state=42)
    return (
        X_single,
        X_single_test,
        X_single_train,
        y_single_test,
        y_single_train,
    )


@app.cell
def _(X_single_train, y_single_train):
    linear_regression_model_single = LinearRegression()
    linear_regression_model_single.fit(X_single_train, y_single_train)
    return (linear_regression_model_single,)


@app.cell
def _(X_single_test, linear_regression_model_single):
    y_single_test_predict = linear_regression_model_single.predict(X_single_test)
    return (y_single_test_predict,)


@app.cell
def _(y_single_test, y_single_test_predict):
    regression_metrics(y_true=y_single_test, y_pred=y_single_test_predict, label='Single Feature Linear Regression')
    return


@app.cell
def _(
    X_single,
    X_single_train,
    linear_regression_model_single,
    y_single_train,
):
    # Visualize the linear fit for the single feature model

    # Create a grid of MedInc values for a smooth line
    X_plot = np.linspace(X_single['MedInc'].min(), X_single['MedInc'].max(), 200).reshape(-1, 1)
    y_plot = linear_regression_model_single.predict(X_plot)

    plt.figure()
    plt.scatter(X_single_train['MedInc'], y_single_train, alpha=0.2, label='Train data')
    plt.plot(X_plot, y_plot, linewidth=2, label='Linear Fit (Degree 1)')
    plt.xlabel('MedInc')
    plt.ylabel('MedHouseVal')
    plt.title('Single Feature Linear Regression (MedInc vs MedHouseVal)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(X_single_test, X_single_train, y_single_test, y_single_train):
    # Comapre polynomial regression models of different degrees on the single feature MedInc

    degrees = [_i for _i in range(1, 100, 2)]
    results = []

    for deg in degrees:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=deg, include_bias=False)),
            ('lin_reg', LinearRegression())
        ])
        model.fit(X_single_train, y_single_train)
        y_single_train_pred = model.predict(X_single_train)
        y_single_test_pred_2 = model.predict(X_single_test)
        mae_train = mean_absolute_error(y_single_train, y_single_train_pred)
        rmse_train = np.sqrt(mean_squared_error(y_single_train, y_single_train_pred))
        r2_train = r2_score(y_single_train, y_single_train_pred)
        mae_test = mean_absolute_error(y_single_test, y_single_test_pred_2)
        rmse_test = np.sqrt(mean_squared_error(y_single_test, y_single_test_pred_2))
        r2_test = r2_score(y_single_test, y_single_test_pred_2)
        results.append({
            'degree': deg,
            'mae_train': mae_train,
            'rmse_train': rmse_train,
            'r2_train': r2_train,
            'mae_test': mae_test,
            'rmse_test': rmse_test,
            'r2_test': r2_test
        })

        results_df = pd.DataFrame(results)
    return (results_df,)


@app.cell
def _(results_df):
    results_df
    return


@app.cell
def _(results_df):
    # Plot R Squared and RMSE vs Polynomial Degree

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # R Squared Plot
    axes[0].plot(results_df['degree'], results_df['r2_train'], marker='o', label='r2 train')
    axes[0].plot(results_df['degree'], results_df['r2_test'], marker='o', label='r2 test')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('R2 Score')
    axes[0].set_title('R2 Score vs Polynomial Degree (Single Feature)')
    axes[0].legend()

    # RMSE Squared Plot
    axes[1].plot(results_df['degree'], results_df['rmse_train'], marker='o', label='rmse train')
    axes[1].plot(results_df['degree'], results_df['rmse_test'], marker='o', label='rmse test')
    axes[1].set_xlabel('Polynomial Degree')
    axes[1].set_ylabel('RMSE Score')
    axes[1].set_title('RMSE Score vs Polynomial Degree (Single Feature)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
