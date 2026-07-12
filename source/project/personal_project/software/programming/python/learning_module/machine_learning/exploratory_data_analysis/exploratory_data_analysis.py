# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "pandas[pyarrow]==2.3.3",
#     "python-lsp-ruff==2.3.0",
#     "python-lsp-server==1.13.2",
#     "pyzmq",
#     "scikit-learn==1.7.2",
#     "seaborn==0.13.2",
#     "vegafusion==2.0.3",
#     "vl-convert-python==1.8.0",
#     "websockets==15.0.1",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")

with app.setup:
    # Initialization code that runs before all other cells

    import numpy as np
    import pandas as pd
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt

    from pathlib import Path
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    # Setting plotting style and seaborn library's font scale

    plt.style.use("default")
    sns.set_theme(font_scale=1.1)


@app.cell
def _():
    filepath = Path('machine_learning_course/exploratory_data_analysis/ecommerce_customer_behavior_dataset_v2.csv')
    df = pd.read_csv(filepath)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    # Quick Descriptive Statistics for Numeric Features

    df.describe().T # Here "T" means transpose
    return


@app.cell
def _(df):
    # Checking Missing Values Count per Column

    df.isnull().sum()
    return


@app.cell
def _():
    # Defining the essential columns with their category

    target_col = "Is_Returning_Customer"
    numerical_cols = ["Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount", "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days", "Customer_Rating"]
    categorical_cols = ["Gender", "City", "Product_Category", "Payment_Method", "Device_Type"]
    return categorical_cols, numerical_cols, target_col


@app.cell
def _(df, numerical_cols):
    # Look at some basic value ranges

    df[numerical_cols].agg(["min", "max", "mean", "median"]).T
    return


@app.cell
def _(df, numerical_cols):
    df[numerical_cols].hist(figsize=(12, 10))
    plt.suptitle("Histogram of Numerical Features")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(df, numerical_cols):
    # Boxplot to get a sense of spread and possible outliers

    plt.figure(figsize=(12, 8))
    df[numerical_cols].boxplot()
    plt.title("Boxplot for Numerical Features")
    plt.xticks(rotation=45)
    plt.show()
    return


@app.class_definition
# Outliers Detection for Numerical Features

class OutlierDetector:

    @staticmethod
    def outliers_count(df: pd.DataFrame, features: list):
        """
        Detects and prints the number of outliers for specified numerical columns
        in a pandas DataFrame using the Interquartile Range (IQR) method.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (list): A list of column names to check for outliers.
        """

        outliers_results = {}

        if len(features) < 1:
            print("Invalid Input!")
            return outliers_results

        for col in features:
            if col not in df.columns:
                print(f"{col} not found in the DataFrame.")
                continue

            quartile_one = df[col].quantile(0.25)
            quartile_three = df[col].quantile(0.75)
            interquartile_range = quartile_three - quartile_one

            lower_range = quartile_one - 1.5 * interquartile_range
            upper_range = quartile_three + 1.5 * interquartile_range

            outliers = (df[col] < lower_range) | (df[col] > upper_range)

            num_outliers = outliers.sum()

            if num_outliers != 0:
                outliers_results[col] = num_outliers
                print(f"Column '{col}':")
                print(f"  - IQR Bounds: [{lower_range:.2f}, {upper_range:.2f}]")
                print(f"  - Number of outliers: {num_outliers}")

        return outliers_results

    @staticmethod
    def outliers_plot(outliers_counts_dict):
        """
        Generates a bar plot of the number of outliers per column.

        Args:
            outlier_counts_dict (dict): A dictionary mapping column names
                                         to their outlier counts.
        """

        if not outliers_counts_dict:
            print('No outliers data to plot.')
            return

        outliers_seris = pd.Series(outliers_counts_dict)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=outliers_seris.index, y=outliers_seris.values)
        plt.title("Count of outliers per column (IQR Method)")
        plt.xlabel('Columns')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.show()


@app.cell
def _(df, numerical_cols):
    counts = OutlierDetector.outliers_count(df, numerical_cols)
    print(counts)
    return (counts,)


@app.cell
def _(counts):
    OutlierDetector.outliers_plot(counts)
    return


@app.cell
def _(numerical_cols):
    class OutliersHandler:

        @staticmethod
        def remove_outliers(df: pd.DataFrame, features: list) -> pd.DataFrame:
            """
            Removes rows from the DataFrame where any of the specified
            numerical columns have an outlier value (based on 1.5*IQR).

            Returns: The DataFrame with outlier rows removed.
            """

            if len(features) < 1:
                print("Invalid Input!")
                return df

            no_outliers_mask = pd.Series(True, index=df.index)

            for col in features:
                if col not in df.columns:
                    print(f"{col} not found in the DataFrame.")
                    continue

                quartile_one = df[col].quantile(0.25)
                quartile_three = df[col].quantile(0.75)
                interquartile_range = quartile_three - quartile_one

                lower_range = quartile_one - 1.5 * interquartile_range
                upper_range = quartile_three + 1.5 * interquartile_range

                col_mask = (df[col] >= lower_range) & (df[col] <= upper_range)

                no_outliers_mask = no_outliers_mask & col_mask
                print(f"  - Column '{col}' bounds: [{lower_range:.2f}, {upper_range:.2f}]")

            df_removed_outliers = df[no_outliers_mask]

            print(f"Removed {len(df) - len(df_removed_outliers)} outlier rows.")
            print(f"New DataFrame shape: {df_removed_outliers.shape}")

            return df_removed_outliers

        @staticmethod
        def cap_outliers(df: pd.DataFrame, features: list) -> pd.DataFrame:
            """
            Caps outliers by setting values below the lower bound to the
            lower bound, and values above the upper bound to the upper bound.

            Returns: A copy of the DataFrame with capped outliers.
            """

            if len(features) < 1:
                print("Invalid Input!")
                return df

            df_capped = df.copy()

            for col in features:
                if col not in df.columns:
                    print(f"{col} not found in the DataFrame.")
                    continue

                quartile_one = df[col].quantile(0.25)
                quartile_three = df[col].quantile(0.75)
                interquartile_range = quartile_three - quartile_one

                lower_range = quartile_one - 1.5 * interquartile_range
                upper_range = quartile_three + 1.5 * interquartile_range

                df_capped[col] = df_capped[col].clip(lower=lower_range, upper=upper_range)

            print(f"  - Column '{col}' capped between: [{lower_range:.2f}, {upper_range:.2f}]")

            return df_capped

        @staticmethod
        def log_transform(df: pd.DataFrame, features: list) -> pd.DataFrame:
            """
            Applies a log(x + 1) transformation (log1p) to specified numerical columns.
            This is suitable for highly skewed data, especially those with values close to zero.

            Args:
                df (pd.DataFrame): The input DataFrame.
                features (list): A list of column names to apply the transformation to.

            Returns:
                pd.DataFrame: A copy of the DataFrame with the specified columns log-transformed.
            """

            if len(features) < 1:
                print("Invalid Input!")
                return df

            df_transformed = df.copy()

            for col in features:
                if col not in df.columns:
                    print(f"{col} not found in the DataFrame.")
                    continue

                if (df_transformed[col] < 0).any():
                    print(f"Column '{col}' contains negative values. Log transformation may not be appropriate or meaningful. Proceeding with log(x + 1) on non-negative part.")

                    # For safety, we set negatives to 0 before log1p,
                    # but this is often better handled by domain knowledge.
                    series_to_transform = df_transformed[col].clip(lower=0)
                else:
                    series_to_transform = df_transformed[col]

                # Use np.log1p for log(x + 1) transformation
                df_transformed[col] = np.log1p(series_to_transform)
                print(f"  - Column '{col}' successfully transformed.")

            return df_transformed

        @staticmethod
        def transformed_box_plot(df: pd.DataFrame, features: list):
            plt.figure(figsize=(12, 8))
            df[numerical_cols].boxplot()
            plt.title("Boxplot for Numerical Features after Transpormation")
            plt.xticks(rotation=45, ha="right")
            plt.show()
    return (OutliersHandler,)


@app.cell
def _(OutliersHandler, df, numerical_cols):
    # Handling outliers using log transpormation

    df_transformed = OutliersHandler.log_transform(df=df, features=numerical_cols)
    df_transformed
    return (df_transformed,)


@app.cell
def _(OutliersHandler, df_transformed, numerical_cols):
    OutliersHandler.transformed_box_plot(df=df_transformed, features=numerical_cols)
    return


@app.cell
def _(numerical_cols):
    class FeatureScaler:
        @staticmethod
        def standard_scaler(df: pd.DataFrame, features: list) -> pd.DataFrame:
            std_scaler = StandardScaler()
            data_to_scale = df[features]
            df_scaled_data = std_scaler.fit_transform(data_to_scale)

            df_scaled = pd.DataFrame(df_scaled_data,
                                     columns=features,
                                     index=df.index)

            return df_scaled

        @staticmethod
        def min_max_scaler(df: pd.DataFrame, features: list) -> pd.DataFrame:
            min_max_scaler = MinMaxScaler()
            data_to_scale = df[features]
            df_scaled_data = min_max_scaler.fit_transform(data_to_scale)

            df_scaled = pd.DataFrame(df_scaled_data,
                                    columns=features,
                                    index=df.index)

            return df_scaled

        @staticmethod
        def robust_scaler(df: pd.DataFrame, features: list) -> pd.DataFrame:
            robust_scaler = RobustScaler()
            data_to_scale = df[features]
            df_scaled_data = robust_scaler.fit_transform(data_to_scale)

            df_scaled = pd.DataFrame(df_scaled_data,
                                    columns=features,
                                    index=df.index)

            return df_scaled

        @staticmethod
        def scaled_box_plot(df: pd.DataFrame, features: list):
            plt.figure(figsize=(12, 8))
            df[numerical_cols].boxplot()
            plt.title("Boxplot for Numerical Features after Transformation")
            plt.xticks(rotation=45, ha="right")
            plt.show()

        @staticmethod
        def scaled_heatmap(df: pd.DataFrame, features: list):
            plt.figure(figsize=(8, 6))
            sns.heatmap(df[numerical_cols].corr(), annot=True, fmt=".2f", linewidths=0.5, cmap="coolwarm", vmin=-1, vmax=1)
            plt.suptitle("Correlation among scaled Numerical Features")
            plt.show()
    return (FeatureScaler,)


@app.cell
def _(FeatureScaler, df_transformed, numerical_cols):
    df_scaled = FeatureScaler.robust_scaler(df=df_transformed, features=numerical_cols)
    df_scaled
    return (df_scaled,)


@app.cell
def _(FeatureScaler, df_scaled, numerical_cols):
    FeatureScaler.scaled_box_plot(df=df_scaled, features=numerical_cols)
    return


@app.cell
def _(df):
    # Drop unnecessary Numerical Features

    df.drop(columns=["Order_ID", "Customer_ID", "Date"], inplace=True)
    return


@app.cell
def _(df, target_col):
    df[target_col] = df[target_col].astype(int)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, target_col):
    # Target Distribution and Imbalance

    plt.figure(figsize=(5, 4))
    sns.countplot(x=df[target_col])
    plt.title("Target Column [Is_Returning_Customer] Distribution")
    plt.xlabel("Is Returning Customer [1: True or 0: False]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(FeatureScaler, df, df_scaled, numerical_cols):
    mo.vstack([
        FeatureScaler.scaled_heatmap(df=df, features=numerical_cols),
        FeatureScaler.scaled_heatmap(df=df_scaled, features=numerical_cols)
    ])
    return


@app.cell
def _(df, target_col):
    df[target_col].value_counts(normalize=True).round(4) * 100
    return


@app.cell
def _(categorical_cols, df):
    # Categorial Feature Exploration

    for _i in categorical_cols:
        plt.figure(figsize=(5, 4))
        df[_i].value_counts().plot(kind="bar")
        plt.title(f"Value Counts for {_i}")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    return


@app.class_definition
class Encoder:
    @staticmethod
    def one_hot_encode(df: pd.DataFrame, categorical_features: list, prefix: str = None, drop_first: bool = False) -> pd.DataFrame:
        df_encoded = pd.get_dummies(
            df,
            columns=categorical_features,
            prefix=prefix,
            dtype=int, # Use integer 0/1 instead of booleans
            drop_first=drop_first # Uses the boolean input directly
        )

        return df_encoded

    @staticmethod
    def label_encode(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
        df_copy = df.copy()
        le = LabelEncoder()

        for col in categorical_features:
            # Apply fit_transform to the specific column
            # .ravel() is used in case the input is a Series that needs flattening
            df_copy[f'{col}_LabelEncoded'] = le.fit_transform(df_copy[col].values)

            df_encoded = df_copy.drop(columns=[col])

            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Encoding map for '{col}': {mapping}")

        return df_encoded


@app.cell
def _(categorical_cols, df):
    for _i in categorical_cols:
        print(f"{_i}: {df[_i].unique()}")
    return


@app.cell
def _(categorical_cols, df_scaled, df_transformed, numerical_cols, target_col):
    # All columns are nominal, so one hot encoding will be applied for all of them

    df_working = df_transformed.copy()
    df_working[numerical_cols] = df_scaled
    df_working.drop(columns=["Order_ID", "Customer_ID", "Date"], inplace=True, errors='ignore')
    df_working[target_col] = df_working[target_col].astype(int)
    df_final = Encoder.one_hot_encode(df=df_working, categorical_features=categorical_cols, drop_first=True)
    return (df_final,)


@app.cell
def _(df_final):
    df_final
    return


if __name__ == "__main__":
    app.run()
