# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "kagglehub==0.3.13",
#     "matplotlib==3.10.8",
#     "numpy==2.4.0",
#     "pandas==2.3.3",
#     "pyarrow==22.0.0",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    # TODO: Import all necessary libraries here

    import kagglehub
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.svm import SVC, SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import RobustScaler, PolynomialFeatures, StandardScaler
    return (
        GaussianNB,
        KNeighborsClassifier,
        LinearRegression,
        LogisticRegression,
        PolynomialFeatures,
        RandomForestClassifier,
        RandomForestRegressor,
        RobustScaler,
        SVC,
        SVR,
        StandardScaler,
        accuracy_score,
        classification_report,
        kagglehub,
        mean_absolute_error,
        mean_squared_error,
        np,
        pd,
        plt,
        r2_score,
        sns,
        train_test_split,
    )


@app.cell
def _(kagglehub):
    _path = kagglehub.dataset_download('mirichoi0218/insurance')
    print('Path to dataset files:', _path)
    return


@app.cell
def _(kagglehub):
    _path = kagglehub.dataset_download('taweilo/loan-approval-classification-data')
    print('Path to dataset files:', _path)
    return


@app.cell
def _(pd):
    # TODO: Load regression dataset

    regression_task_df = pd.read_csv("/home/abdullahalazmi/.cache/kagglehub/datasets/mirichoi0218/insurance/versions/1/insurance.csv")

    regression_task_df.head(5)
    return (regression_task_df,)


@app.cell
def _(regression_task_df):
    print(f"Dataset Shape: {regression_task_df.shape}")
    return


@app.cell
def _(regression_task_df):
    # TODO: Perform EDA

    regression_task_df.describe().T
    return


@app.cell
def _(regression_task_df):
    regression_task_df.isnull().sum()
    return


@app.cell
def _(regression_task_df):
    regression_task_df.columns.tolist()
    return


@app.cell
def _(regression_task_df):
    print('Column: Unique Values')
    for _i in ['sex', 'smoker', 'region']:
        print(f'{_i}: {regression_task_df[_i].unique()}')
    return


@app.cell
def _(pd, regression_task_df):
    regression_task_df['sex'] = regression_task_df['sex'].map({'female': 0, 'male': 1})
    regression_task_df['smoker'] = regression_task_df['smoker'].map({'no': 0, 'yes': 1})
    regression_task_df_1 = pd.get_dummies(regression_task_df, columns=['region'], drop_first=True, dtype=int)
    return (regression_task_df_1,)


@app.cell
def _(regression_task_df_1):
    regression_task_df_1.head(5)
    return


@app.cell
def _(regression_task_df_1):
    reg_corr = regression_task_df_1.corr()
    return (reg_corr,)


@app.cell
def _(plt, reg_corr, sns):
    plt.figure(figsize=(10, 6))

    sns.heatmap(reg_corr, vmin=-1, vmax=1, cmap='coolwarm')

    plt.suptitle('Correlation among Features of Dataset for Regression Task')
    plt.show()
    return


@app.cell
def _(plt, regression_task_df_1, sns):
    # Feature vs class plot
    plt.figure(figsize=(10, 6))
    sns.regplot(data=regression_task_df_1, x='charges', y='age', scatter=True, fit_reg=True)
    plt.title('Charges vs Age')
    plt.xlabel('Charges')
    plt.ylabel('Age')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(plt, regression_task_df_1, sns):
    # Feature vs class plot
    plt.figure(figsize=(10, 6))
    sns.regplot(data=regression_task_df_1, x='charges', y='bmi', scatter=True, fit_reg=True)
    plt.title('Charges vs BMI')
    plt.xlabel('Charges')
    plt.ylabel('BMI')
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(RobustScaler, regression_task_df_1, train_test_split):
    # TODO: Prepare features
    X_reg = regression_task_df_1.drop('charges', axis=1)
    y_reg = regression_task_df_1['charges']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, random_state=42, test_size=0.25)
    scaler_reg = RobustScaler()
    X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
    X_test_reg_scaled = scaler_reg.transform(X_test_reg)
    return (
        X_test_reg_scaled,
        X_train_reg,
        X_train_reg_scaled,
        y_test_reg,
        y_train_reg,
    )


@app.cell
def _(LinearRegression, X_test_reg_scaled, X_train_reg_scaled, y_train_reg):
    # TODO: Multiple Linear Regression

    lr_model = LinearRegression()
    lr_model.fit(X_train_reg_scaled, y_train_reg)

    y_train_pred_lr = lr_model.predict(X_train_reg_scaled)
    y_test_pred_lr = lr_model.predict(X_test_reg_scaled)
    return y_test_pred_lr, y_train_pred_lr


@app.cell
def _(mean_absolute_error, mean_squared_error, np, pd, r2_score):
    def regression_metrics(y_true, y_pred, condition='Test', model=None):
        res = []
    
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
    
        res.append({f'{model} MAE {condition}': mae, f'{model} RMSE {condition}': rmse, f'{model} R2 {condition}': r2})
    
        return pd.DataFrame(res)
    return (regression_metrics,)


@app.cell
def _(regression_metrics, y_train_pred_lr, y_train_reg):
    regression_metrics(y_train_reg, y_train_pred_lr, condition='Train', model='MLR')
    return


@app.cell
def _(regression_metrics, y_test_pred_lr, y_test_reg):
    regression_metrics(y_test_reg, y_test_pred_lr, condition='Test', model='MLR')
    return


@app.cell
def _(PolynomialFeatures):
    poly_features = {
        'PF_01': PolynomialFeatures(degree=2),
        'PF_02': PolynomialFeatures(degree=3),
        'PF_03': PolynomialFeatures(degree=4),
        'PF_04': PolynomialFeatures(degree=5),
        'PF_05': PolynomialFeatures(degree=6),
        'PF_06': PolynomialFeatures(degree=7),
        'PF_07': PolynomialFeatures(degree=8),
        'PF_08': PolynomialFeatures(degree=9),
        'PF_09': PolynomialFeatures(degree=10),
        'PF_10': PolynomialFeatures(degree=11)
    }
    return (poly_features,)


@app.cell
def _(
    LinearRegression,
    X_test_reg_scaled,
    X_train_reg_scaled,
    poly_features,
    regression_metrics,
    y_test_reg,
    y_train_reg,
):
    poly_train_results = []
    poly_test_results = []
    for _name, feature in poly_features.items():
        X_train_reg_poly = feature.fit_transform(X_train_reg_scaled)
        X_test_reg_poly = feature.transform(X_test_reg_scaled)
    
        poly_model = LinearRegression()
        poly_model.fit(X_train_reg_poly, y_train_reg)
    
        y_train_pred_poly = poly_model.predict(X_train_reg_poly)
        y_test_pred_poly = poly_model.predict(X_test_reg_poly)
    
        _train_metrics = regression_metrics(y_train_reg, y_train_pred_poly, condition='Train', model='PR')
        _test_metrics = regression_metrics(y_test_reg, y_test_pred_poly, condition='Test', model='PR')
    
        poly_train_results.append(_train_metrics)
        poly_test_results.append(_test_metrics)
    return poly_test_results, poly_train_results


@app.cell
def _(pd, poly_train_results):
    pr_train_df = pd.concat(poly_train_results, ignore_index=True)
    pr_train_df
    return (pr_train_df,)


@app.cell
def _(pd, poly_test_results):
    pr_test_df = pd.concat(poly_test_results, ignore_index=True)
    pr_test_df
    return (pr_test_df,)


@app.cell
def _(plt, pr_train_df):
    plt.figure(figsize=(10, 6))

    plt.plot(pr_train_df.index + 2, pr_train_df['PR MAE Train'], label="PR MAE Train", marker='o')
    plt.plot(pr_train_df.index + 2, pr_train_df['PR RMSE Train'], label="PR RMSE Train", marker='s')
    plt.plot(pr_train_df.index + 2, pr_train_df['PR R2 Train'], label="PR R2 Train", marker='o')

    plt.xlabel('Degree')
    plt.ylabel('Error Matrix Results')

    plt.title('Polynomial Regression Train Metrics by Degree')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, pr_test_df):
    plt.figure(figsize=(10, 6))

    plt.plot(pr_test_df.index + 2, pr_test_df['PR MAE Test'], label="PR MAE Test", marker='o')
    plt.plot(pr_test_df.index + 2, pr_test_df['PR RMSE Test'], label="PR RMSE Test", marker='s')
    plt.plot(pr_test_df.index + 2, pr_test_df['PR R2 Test'], label="PR MAE R2", marker='o')

    plt.xlabel('Degree')
    plt.ylabel('Error Matrix Results')

    plt.title('Polynomial Regression Test Metrics by Degree')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(SVR):
    # TODO: Support Vector Regression

    svr_models = {
        'SVR_RBF_01': SVR(kernel='rbf', gamma='scale', C=100.0, epsilon=0.1),
        'SVR_RBF_02': SVR(kernel='rbf', gamma='scale', C=90.0, epsilon=0.2),
        'SVR_RBF_03': SVR(kernel='rbf', gamma='scale', C=80.0, epsilon=0.3),
        'SVR_RBF_04': SVR(kernel='rbf', gamma='scale', C=70.0, epsilon=0.4),
        'SVR_RBF_05': SVR(kernel='rbf', gamma='scale', C=60.0, epsilon=0.5),
        'SVR_RBF_06': SVR(kernel='rbf', gamma='scale', C=50.0, epsilon=0.6),
        'SVR_RBF_07': SVR(kernel='rbf', gamma='scale', C=40.0, epsilon=0.7),
        'SVR_RBF_08': SVR(kernel='rbf', gamma='scale', C=30.0, epsilon=0.8),
        'SVR_RBF_09': SVR(kernel='rbf', gamma='scale', C=20.0, epsilon=0.9),
        'SVR_RBF_10': SVR(kernel='rbf', gamma='scale', C=10.0, epsilon=1.0),
    }
    return (svr_models,)


@app.cell
def _(
    X_test_reg_scaled,
    X_train_reg_scaled,
    regression_metrics,
    svr_models,
    y_test_reg,
    y_train_reg,
):
    svr_train_results = []
    svr_test_results = []
    for _name, _model in svr_models.items():
        _model.fit(X_train_reg_scaled, y_train_reg)
    
        y_train_pred_svr = _model.predict(X_train_reg_scaled)
        y_test_pred_svr = _model.predict(X_test_reg_scaled)
    
        _train_metrics = regression_metrics(y_train_reg, y_train_pred_svr, condition='Train', model='SVR (Kernel: RBF)')
        _test_metrics = regression_metrics(y_test_reg, y_test_pred_svr, condition='Test', model='SVR (Kernel: RBF)')
    
        svr_train_results.append(_train_metrics)
        svr_test_results.append(_test_metrics)
    return svr_test_results, svr_train_results


@app.cell
def _(pd, svr_train_results):
    svr_train_df = pd.concat(svr_train_results, ignore_index=True)
    svr_train_df
    return (svr_train_df,)


@app.cell
def _(pd, svr_test_results):
    svr_test_df = pd.concat(svr_test_results, ignore_index=True)
    svr_test_df
    return (svr_test_df,)


@app.cell
def _(plt, svr_train_df):
    plt.figure(figsize=(10, 6))

    plt.plot(svr_train_df.index + 1, svr_train_df['SVR (Kernel: RBF) MAE Train'], label="SVR (Kernel: RBF) MAE Train", marker='o')
    plt.plot(svr_train_df.index + 1, svr_train_df['SVR (Kernel: RBF) RMSE Train'], label="SVR (Kernel: RBF) RMSE Train", marker='s')
    plt.plot(svr_train_df.index + 1, svr_train_df['SVR (Kernel: RBF) R2 Train'], label="SVR (Kernel: RBF) R2 Train", marker='o')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Regression (Kernel: RBF) Train Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, svr_test_df):
    plt.figure(figsize=(10, 6))

    plt.plot(svr_test_df.index + 1, svr_test_df['SVR (Kernel: RBF) MAE Test'], label="SVR (Kernel: RBF) MAE Test", marker='o')
    plt.plot(svr_test_df.index + 1, svr_test_df['SVR (Kernel: RBF) RMSE Test'], label="SVR (Kernel: RBF) RMSE Test", marker='s')
    plt.plot(svr_test_df.index + 1, svr_test_df['SVR (Kernel: RBF) R2 Test'], label="SVR (Kernel: RBF) R2 Test", marker='o')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Regression (Kernel: RBF) Test Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(RandomForestRegressor):
    # TODO: Random Forest Regressor

    rfr_models = {
        'RFR_01': RandomForestRegressor(n_estimators=10, random_state=42),
        'RFR_02': RandomForestRegressor(n_estimators=20, random_state=42),
        'RFR_03': RandomForestRegressor(n_estimators=30, random_state=42),
        'RFR_04': RandomForestRegressor(n_estimators=40, random_state=42),
        'RFR_05': RandomForestRegressor(n_estimators=50, random_state=42),
        'RFR_06': RandomForestRegressor(n_estimators=60, random_state=42),
        'RFR_07': RandomForestRegressor(n_estimators=70, random_state=42),
        'RFR_08': RandomForestRegressor(n_estimators=80, random_state=42),
        'RFR_09': RandomForestRegressor(n_estimators=90, random_state=42),
        'RFR_10': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    return (rfr_models,)


@app.cell
def _(
    X_test_reg_scaled,
    X_train_reg,
    X_train_reg_scaled,
    regression_metrics,
    rfr_models,
    y_test_reg,
    y_train_reg,
):
    rfr_train_results = []
    rfr_test_results = []
    for _name, _model in rfr_models.items():
        _model.fit(X_train_reg, y_train_reg)
    
        y_train_pred_rfr = _model.predict(X_train_reg_scaled)
        y_test_pred_rfr = _model.predict(X_test_reg_scaled)
    
        _train_metrics = regression_metrics(y_train_reg, y_train_pred_rfr, condition='Train', model='RFR')
        _test_metrics = regression_metrics(y_test_reg, y_test_pred_rfr, condition='Test', model='RFR')
    
        rfr_train_results.append(_train_metrics)
        rfr_test_results.append(_test_metrics)
    return rfr_test_results, rfr_train_results


@app.cell
def _(pd, rfr_train_results):
    rfr_train_df = pd.concat(rfr_train_results, ignore_index=True)
    rfr_train_df
    return (rfr_train_df,)


@app.cell
def _(pd, rfr_test_results):
    rfr_test_df = pd.concat(rfr_test_results, ignore_index=True)
    rfr_test_df
    return (rfr_test_df,)


@app.cell
def _(plt, rfr_train_df):
    plt.figure(figsize=(10, 6))

    plt.plot(rfr_train_df.index + 1, rfr_train_df['RFR MAE Train'], label="RFR MAE Train", marker='o')
    plt.plot(rfr_train_df.index + 1, rfr_train_df['RFR RMSE Train'], label="RFR RMSE Train", marker='s')
    plt.plot(rfr_train_df.index + 1, rfr_train_df['RFR R2 Train'], label="RFR R2 Train", marker='o')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Regression Train Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, rfr_test_df):
    plt.figure(figsize=(10, 6))

    plt.plot(rfr_test_df.index + 1, rfr_test_df['RFR MAE Test'], label="RFR MAE Test", marker='o')
    plt.plot(rfr_test_df.index + 1, rfr_test_df['RFR RMSE Test'], label="RFR RMSE Test", marker='s')
    plt.plot(rfr_test_df.index + 1, rfr_test_df['RFR R2 Test'], label="RFR R2 Test", marker='o')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Regression Test Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(pd):
    # TODO: Load classification dataset

    classification_task_df = pd.read_csv("/home/abdullahalazmi/.cache/kagglehub/datasets/taweilo/loan-approval-classification-data/versions/1/loan_data.csv")
    classification_task_df.head(5)
    return (classification_task_df,)


@app.cell
def _(classification_task_df):
    classification_task_df.isnull().sum()
    return


@app.cell
def _(classification_task_df):
    classification_task_df.columns.to_list()
    return


@app.cell
def _(classification_task_df):
    print('Column: Unique Values')
    for _i in ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']:
        print(f'{_i}: {classification_task_df[_i].unique()}')
    return


@app.cell
def _(classification_task_df):
    classification_task_df['person_gender'] = classification_task_df['person_gender'].map({'female': 0, 'male': 1})
    classification_task_df['person_education'] = classification_task_df['person_education'].map({'High School': 0, 'Bachelor': 1, 'Master': 2, 'Doctorate': 3, 'Associate': 4})
    classification_task_df['previous_loan_defaults_on_file'] = classification_task_df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
    return


@app.cell
def _(pd):
    dummies = pd.get_dummies(classification_task_df[['person_home_ownership', 'loan_intent']], drop_first=True, dtype=int)

    classification_task_df = pd.concat([classification_task_df.drop(['person_home_ownership', 'loan_intent'], axis=1), dummies], axis=1)
    return (classification_task_df,)


@app.cell
def _(classification_task_df):
    classification_task_df.head(5)
    return


@app.cell
def _(classification_task_df):
    clf_corr = classification_task_df.corr()
    return (clf_corr,)


@app.cell
def _(clf_corr, plt, sns):
    plt.figure(figsize=(10, 6))

    sns.heatmap(clf_corr, vmin=-1, vmax=1, cmap='coolwarm')

    plt.suptitle('Correlation among features in classification dataset')
    plt.show()
    return


@app.cell
def _(classification_task_df, plt, sns):
    # TODO: Classification EDA
    plt.figure(figsize=(10, 6))

    sns.countplot(x=classification_task_df['loan_status'])

    plt.title('Class Distribution of Target Feature (loan_status)')
    plt.xlabel('Loan Status (0 = Not Approved, 1 = Approved)')
    plt.ylabel('Count')

    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(classification_task_df, plt, sns):
    # Feature vs class plot
    plt.figure(figsize=(10, 6))

    sns.regplot(data=classification_task_df, x='person_income', y='loan_status', scatter=True)

    plt.title('Interest Rate vs Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Interest Rate')
    plt.show()
    return


@app.cell
def _(StandardScaler, classification_task_df, train_test_split):
    # TODO: Prepare classification features
    X_clf = classification_task_df.drop('loan_status', axis=1)
    y_clf = classification_task_df['loan_status']

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

    scaler_clf = StandardScaler()

    X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler_clf.transform(X_test_clf)
    return (
        X_test_clf,
        X_test_clf_scaled,
        X_train_clf,
        X_train_clf_scaled,
        y_test_clf,
        y_train_clf,
    )


@app.cell
def _(accuracy_score, classification_report, pd):
    def classification_metrics(y_true, y_pred, condition='Test', model='Model'):
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
    
        res = {
            f'{condition} Accuracy {model}': [acc], 
            f'{condition} Precision (weighted) {model}': [report['weighted avg']['precision']], 
            f'{condition} Recall (weighted) {model}': [report['weighted avg']['recall']], 
            f'{condition} F1-Score (weighted) {model}': [report['weighted avg']['f1-score']]
        }
    
        return pd.DataFrame(res)
    return (classification_metrics,)


@app.cell
def _(LogisticRegression, X_train_clf_scaled, y_train_clf):
    # TODO: Logistic Regression

    log_reg = LogisticRegression(random_state=42)

    log_reg.fit(X_train_clf_scaled, y_train_clf)
    return (log_reg,)


@app.cell
def _(X_test_clf_scaled, X_train_clf_scaled, log_reg):
    y_train_pred_log_reg = log_reg.predict(X_train_clf_scaled)
    y_test_pred_log_reg = log_reg.predict(X_test_clf_scaled)
    return y_test_pred_log_reg, y_train_pred_log_reg


@app.cell
def _(classification_metrics, y_train_clf, y_train_pred_log_reg):
    classification_metrics(y_train_clf, y_train_pred_log_reg, condition='Train', model='LR')
    return


@app.cell
def _(classification_metrics, y_test_clf, y_test_pred_log_reg):
    classification_metrics(y_test_clf, y_test_pred_log_reg, condition='Test', model='LR')
    return


@app.cell
def _(SVC):
    svc_models = {
        'SVC_RBF_01': SVC(kernel='rbf', gamma='scale', random_state=42, C=100.0),
        'SVC_RBF_02': SVC(kernel='rbf', gamma='scale', random_state=42, C=90.0),
        'SVC_RBF_03': SVC(kernel='rbf', gamma='scale', random_state=42, C=80.0),
        'SVC_RBF_04': SVC(kernel='rbf', gamma='scale', random_state=42, C=70.0),
        'SVC_RBF_05': SVC(kernel='rbf', gamma='scale', random_state=42, C=60.0),
        'SVC_RBF_06': SVC(kernel='rbf', gamma='scale', random_state=42, C=50.0),
        'SVC_RBF_07': SVC(kernel='rbf', gamma='scale', random_state=42, C=40.0),
        'SVC_RBF_08': SVC(kernel='rbf', gamma='scale', random_state=42, C=30.0),
        'SVC_RBF_09': SVC(kernel='rbf', gamma='scale', random_state=42, C=20.0),
        'SVC_RBF_10': SVC(kernel='rbf', gamma='scale', random_state=42, C=10.0),
    }
    return (svc_models,)


@app.cell
def _(
    X_test_clf_scaled,
    X_train_clf_scaled,
    classification_metrics,
    svc_models,
    y_test_clf,
    y_train_clf,
):
    # TODO: SVM Classification
    svc_train_results = []
    svc_test_results = []
    for _name, _model in svc_models.items():
        _model.fit(X_train_clf_scaled, y_train_clf)
    
        y_train_pred_svc = _model.predict(X_train_clf_scaled)
        y_test_pred_svc = _model.predict(X_test_clf_scaled)
    
        _train_metrics = classification_metrics(y_train_clf, y_train_pred_svc, condition='Train', model='SVC (Kernel: RBF)')
        _test_metrics = classification_metrics(y_test_clf, y_test_pred_svc, condition='Test', model='SVC (Kernel: RBF)')
    
        svc_train_results.append(_train_metrics)
        svc_test_results.append(_test_metrics)
    return svc_test_results, svc_train_results


@app.cell
def _(pd, svc_train_results):
    svc_train_df = pd.concat(svc_train_results, ignore_index=True)
    svc_train_df
    return (svc_train_df,)


@app.cell
def _(pd, svc_test_results):
    svc_test_df = pd.concat(svc_test_results, ignore_index=True)
    svc_test_df
    return (svc_test_df,)


@app.cell
def _(plt, svc_train_df):
    plt.figure(figsize=(10, 6))

    plt.plot(svc_train_df.index + 1, svc_train_df['Train Accuracy SVC (Kernel: RBF)'], label="Train Accuracy SVC (Kernel: RBF)", marker='o')
    plt.plot(svc_train_df.index + 1, svc_train_df['Train Precision (weighted) SVC (Kernel: RBF)'], label="Train Precision (weighted) SVC (Kernel: RBF)", marker='s')
    plt.plot(svc_train_df.index + 1, svc_train_df['Train Recall (weighted) SVC (Kernel: RBF)'], label="Train Recall (weighted) SVC (Kernel: RBF)", marker='o')
    plt.plot(svc_train_df.index + 1, svc_train_df['Train F1-Score (weighted) SVC (Kernel: RBF)'], label="Train F1-Score (weighted) SVC (Kernel: RBF)", marker='s')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Classification (Kernel: RBF) Train Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, svc_test_df):
    plt.figure(figsize=(10, 6))

    plt.plot(svc_test_df.index + 1, svc_test_df['Test Accuracy SVC (Kernel: RBF)'], label="Test Accuracy SVC (Kernel: RBF)", marker='o')
    plt.plot(svc_test_df.index + 1, svc_test_df['Test Precision (weighted) SVC (Kernel: RBF)'], label="Test Precision (weighted) SVC (Kernel: RBF)", marker='s')
    plt.plot(svc_test_df.index + 1, svc_test_df['Test Recall (weighted) SVC (Kernel: RBF)'], label="Test Recall (weighted) SVC (Kernel: RBF)", marker='o')
    plt.plot(svc_test_df.index + 1, svc_test_df['Test F1-Score (weighted) SVC (Kernel: RBF)'], label="Test F1-Score (weighted) SVC (Kernel: RBF)", marker='o')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Classification (Kernel: RBF) Test Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(GaussianNB, X_train_clf, y_train_clf):
    # TODO: Naive Bayes

    gnb = GaussianNB()
    gnb.fit(X_train_clf, y_train_clf)
    return (gnb,)


@app.cell
def _(X_test_clf, X_train_clf, gnb):
    y_train_pred_gnb = gnb.predict(X_train_clf)
    y_test_pred_gnb = gnb.predict(X_test_clf)
    return y_test_pred_gnb, y_train_pred_gnb


@app.cell
def _(classification_metrics, y_train_clf, y_train_pred_gnb):
    classification_metrics(y_train_clf, y_train_pred_gnb, condition='Train', model='GNB')
    return


@app.cell
def _(classification_metrics, y_test_clf, y_test_pred_gnb):
    classification_metrics(y_test_clf, y_test_pred_gnb, condition='Test', model='GNB')
    return


@app.cell
def _(KNeighborsClassifier):
    # TODO: KNN

    knn_models = {
        'KNN_01': KNeighborsClassifier(n_neighbors=100),
        'KNN_02': KNeighborsClassifier(n_neighbors=90),
        'KNN_03': KNeighborsClassifier(n_neighbors=80),
        'KNN_04': KNeighborsClassifier(n_neighbors=70),
        'KNN_05': KNeighborsClassifier(n_neighbors=60),
        'KNN_06': KNeighborsClassifier(n_neighbors=50),
        'KNN_07': KNeighborsClassifier(n_neighbors=40),
        'KNN_08': KNeighborsClassifier(n_neighbors=30),
        'KNN_09': KNeighborsClassifier(n_neighbors=20),
        'KNN_10': KNeighborsClassifier(n_neighbors=10)
    }
    return (knn_models,)


@app.cell
def _(
    X_test_clf_scaled,
    X_train_clf_scaled,
    classification_metrics,
    knn_models,
    y_test_clf,
    y_train_clf,
):
    # TODO: SVM Classification
    knn_train_results = []
    knn_test_results = []
    for _name, _model in knn_models.items():
        _model.fit(X_train_clf_scaled, y_train_clf)
    
        y_train_pred_knn = _model.predict(X_train_clf_scaled)
        y_test_pred_knn = _model.predict(X_test_clf_scaled)
    
        _train_metrics = classification_metrics(y_train_clf, y_train_pred_knn, condition='Train', model='KNN')
        _test_metrics = classification_metrics(y_test_clf, y_test_pred_knn, condition='Test', model='KNN')
    
        knn_train_results.append(_train_metrics)
        knn_test_results.append(_test_metrics)
    return knn_test_results, knn_train_results


@app.cell
def _(knn_train_results, pd):
    knn_train_df = pd.concat(knn_train_results, ignore_index=True)
    knn_train_df
    return (knn_train_df,)


@app.cell
def _(knn_test_results, pd):
    knn_test_df = pd.concat(knn_test_results, ignore_index=True)
    knn_test_df
    return (knn_test_df,)


@app.cell
def _(knn_train_df, plt):
    plt.figure(figsize=(10, 6))

    plt.plot(knn_train_df.index + 1, knn_train_df['Train Accuracy KNN'], label="Train Accuracy KNN", marker='o')
    plt.plot(knn_train_df.index + 1, knn_train_df['Train Precision (weighted) KNN'], label="Train Precision (weighted) KNN", marker='s')
    plt.plot(knn_train_df.index + 1, knn_train_df['Train Recall (weighted) KNN'], label="Train Recall (weighted) KNN", marker='o')
    plt.plot(knn_train_df.index + 1, knn_train_df['Train F1-Score (weighted) KNN'], label="Train F1-Score (weighted) KNN", marker='s')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Classification Train Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(knn_test_df, plt):
    plt.figure(figsize=(10, 6))

    plt.plot(knn_test_df.index + 1, knn_test_df['Test Accuracy KNN'], label="Test Accuracy KNN", marker='o')
    plt.plot(knn_test_df.index + 1, knn_test_df['Test Precision (weighted) KNN'], label="Test Precision (weighted) KNN", marker='s')
    plt.plot(knn_test_df.index + 1, knn_test_df['Test Recall (weighted) KNN'], label="Test Recall (weighted) KNN", marker='o')
    plt.plot(knn_test_df.index + 1, knn_test_df['Test F1-Score (weighted) KNN'], label="Test F1-Score (weighted) KNN", marker='s')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Support Vector Classification Test Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(RandomForestClassifier):
    # TODO: Random Forest Classifier

    rfc_models = {
        'RFC_01': RandomForestClassifier(n_estimators=10, random_state=42),
        'RFC_02': RandomForestClassifier(n_estimators=20, random_state=42),
        'RFC_03': RandomForestClassifier(n_estimators=30, random_state=42),
        'RFC_04': RandomForestClassifier(n_estimators=40, random_state=42),
        'RFC_05': RandomForestClassifier(n_estimators=50, random_state=42),
        'RFC_06': RandomForestClassifier(n_estimators=60, random_state=42),
        'RFC_07': RandomForestClassifier(n_estimators=70, random_state=42),
        'RFC_08': RandomForestClassifier(n_estimators=80, random_state=42),
        'RFC_09': RandomForestClassifier(n_estimators=90, random_state=42),
        'RFC_10': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    return (rfc_models,)


@app.cell
def _(
    X_test_clf_scaled,
    X_train_clf_scaled,
    classification_metrics,
    rfc_models,
    y_test_clf,
    y_train_clf,
):
    rfc_train_results = []
    rfc_test_results = []
    for _name, _model in rfc_models.items():
        _model.fit(X_train_clf_scaled, y_train_clf)
    
        y_train_pred_rfc = _model.predict(X_train_clf_scaled)
        y_test_pred_rfc = _model.predict(X_test_clf_scaled)
    
        _train_metrics = classification_metrics(y_train_clf, y_train_pred_rfc, condition='Train', model='RFC')
        _test_metrics = classification_metrics(y_test_clf, y_test_pred_rfc, condition='Test', model='RFC')
    
        rfc_train_results.append(_train_metrics)
        rfc_test_results.append(_test_metrics)
    return rfc_test_results, rfc_train_results


@app.cell
def _(pd, rfc_train_results):
    rfc_train_df = pd.concat(rfc_train_results, ignore_index=True)
    rfc_train_df
    return (rfc_train_df,)


@app.cell
def _(pd, rfc_test_results):
    rfc_test_df = pd.concat(rfc_test_results, ignore_index=True)
    rfc_test_df
    return (rfc_test_df,)


@app.cell
def _(plt, rfc_train_df):
    plt.figure(figsize=(10, 6))

    plt.plot(rfc_train_df.index + 1, rfc_train_df['Train Accuracy RFC'], label="Train Accuracy RFC", marker='o')
    plt.plot(rfc_train_df.index + 1, rfc_train_df['Train Precision (weighted) RFC'], label="Train Precision (weighted) RFC", marker='s')
    plt.plot(rfc_train_df.index + 1, rfc_train_df['Train Recall (weighted) RFC'], label="Train Recall (weighted) RFC", marker='o')
    plt.plot(rfc_train_df.index + 1, rfc_train_df['Train F1-Score (weighted) RFC'], label="Train F1-Score (weighted) RFC", marker='s')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Random Forest Classification Train Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


@app.cell
def _(plt, rfc_test_df):
    plt.figure(figsize=(10, 6))

    plt.plot(rfc_test_df.index + 1, rfc_test_df['Test Accuracy RFC'], label="Test Accuracy RFC", marker='o')
    plt.plot(rfc_test_df.index + 1, rfc_test_df['Test Precision (weighted) RFC'], label="Test Precision (weighted) RFC", marker='s')
    plt.plot(rfc_test_df.index + 1, rfc_test_df['Test Recall (weighted) RFC'], label="Test Recall (weighted) RFC", marker='o')
    plt.plot(rfc_test_df.index + 1, rfc_test_df['Test F1-Score (weighted) RFC'], label="Test F1-Score (weighted) RFC", marker='s')

    plt.xlabel('Index')
    plt.ylabel('Error Matrix Results')

    plt.title('Random Forest Classification Test Metrics')
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
