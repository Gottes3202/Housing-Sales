import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
    from sklearn.feature_selection import mutual_info_regression, SelectKBest
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler, FunctionTransformer
    from sklearn.metrics import mean_absolute_error
    from xgboost import XGBRegressor
    from xgboost.callback import EarlyStopping
    from sklearn.pipeline import Pipeline
    import marimo as mo
    from sklearn.cluster import KMeans
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Continuous, Integer
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    return (
        ColumnTransformer,
        KFold,
        OrdinalEncoder,
        Pipeline,
        SimpleImputer,
        XGBRegressor,
        cross_val_score,
        mean_absolute_error,
        mo,
        pd,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md("""
    # Housing model
    """).center()
    return


@app.cell
def data_import(pd):
    #Импорт данных
    X_data = pd.read_csv('D:/VSProjects/Jupyter_Projects/Kaggle/train.csv', index_col='Id')
    X_test = pd.read_csv('D:/VSProjects/Jupyter_Projects/Kaggle/test.csv', index_col='Id')
    return X_data, X_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Исходные данные
    """)
    return


@app.cell
def dataframe_visualisation(X_data, mo):
    mo.plain(X_data)
    return


@app.cell
def data_preprocesssing(X_data, train_test_split):
    X_data.dropna(subset=['SalePrice'], inplace=True, axis=0)
    y_data = X_data.SalePrice.copy()
    X_data.drop(['SalePrice'], inplace=True, axis=1)
    X_train, X_test1, y_train, y_test1 = train_test_split(X_data, y_data, train_size=0.7, test_size=0.3, random_state=0)
    return X_test1, X_train, y_data, y_test1, y_train


@app.cell
def _(
    ColumnTransformer,
    OrdinalEncoder,
    Pipeline,
    SimpleImputer,
    XGBRegressor,
    X_data,
    X_train,
):
    num_cols = [cname for cname in X_train.columns if X_data[cname].dtype in ['int64', 'float64']]
    cat_cols = [cname for cname in X_train.columns if X_data[cname].dtype in ['object']]

    preprocessor = ColumnTransformer(transformers=[
        ('num', SimpleImputer(strategy='most_frequent'), num_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), cat_cols)
    ])

    regressor = XGBRegressor(
        random_state=0,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=3,
        objective='reg:absoluteerror'
    )

    model = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('modeling', regressor)
    ])
    return (model,)


@app.cell
def model_fitting_and_validation(
    KFold,
    X_train,
    cross_val_score,
    model,
    y_train,
):
    #Обучение и валидация модели
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    scores = cross_val_score(
        estimator=model,
        X=X_train, y=y_train,
        scoring='neg_mean_absolute_error',
        cv=kf,
        n_jobs=-1
    )

    MAE = -1*scores.mean()
    print(f'Средняя ошибка: {MAE:.2f}')
    return


@app.cell
def _(X_test1, X_train, mean_absolute_error, model, y_test1, y_train):
    #test
    y_pred = model.fit(X_train, y_train).predict(X_test1)
    test_mae = mean_absolute_error(y_test1, y_pred)

    print(f'Ошибка на тестовых данных: {test_mae:.2f}')

    return


@app.cell(disabled=True)
def test_for_competition(X_data, X_test, grid, pd, y_data):
    grid.best_estimator_.fit(X_data, y_data)
    preds_test = grid.best_estimator_.predict(X_test)
    output = pd.DataFrame({'Id': X_test.index,
                           'SalePrice': preds_test})
    output.to_csv('D:/VSProjects/Jupyter_Projects/Kaggle/submission.csv', index=False)
    return


if __name__ == "__main__":
    app.run()
