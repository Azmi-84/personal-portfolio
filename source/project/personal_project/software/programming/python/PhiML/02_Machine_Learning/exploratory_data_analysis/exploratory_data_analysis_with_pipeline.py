# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "scikit-learn==1.7.2",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells

    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler


@app.cell
def _():
    df = pd.read_csv("01_Exploratory_Data_Analysis/ecommerce_customer_behavior_dataset_v2.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    # We've completed the Exploratory Data Analysis in the previous notebook and this notebook is for pipeline

    # For Numerical Features we are applying the (Log Transpormation + Robust Scaling)

    class log1pTransformer():
        def fit(self, X, y=None):
            return self
        def transform(self, X, y=None):
            return np.log1p(X.clip(lower=0))

    numerical_transformer = Pipeline(steps=[
        ('log', log1pTransformer()),
        ('robust', RobustScaler())
    ])
    return (numerical_transformer,)


@app.cell
def _():
    # Categorial Transformer (One-Hot Encoding)

    categorial_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    return (categorial_transformer,)


@app.cell
def _(
    categorial_transformer,
    categorical_cols,
    numerical_cols,
    numerical_transformer,
):
    # Column Transformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_transformer, numerical_cols),
            ('categorical', categorial_transformer, categorical_cols),
            ('drop_cols', 'drop', ["Order_ID", "Customer_ID", "Date"])
        ],
        remainder='passthrough'
    )
    return (preprocessor,)


@app.cell
def _():
    target_column = "Is_Returning_Customer"
    numerical_cols = ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount", "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"]
    categorical_cols = ["Gender", "City", "Product_Category", "Payment_Method", "Device_Type"]
    return categorical_cols, numerical_cols, target_column


@app.cell
def _(df, target_column):
    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, X_train, preprocessor):
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    return (X_train_processed,)


@app.cell
def _(X_train, X_train_processed):
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Processed X_train shape: {X_train_processed.shape}")
    return


@app.cell
def _(preprocessor):
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return (full_pipeline,)


@app.cell
def _(X_train, full_pipeline, y_train):
    full_pipeline.fit(X_train, y_train)
    return


@app.cell
def _(X_test, full_pipeline):
    y_pred = full_pipeline.predict(X_test)
    return (y_pred,)


@app.cell
def _(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy on Test Set: {accuracy:.4f}")
    return


if __name__ == "__main__":
    app.run()
