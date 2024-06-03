import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import os


def load_data():
    base_path = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(base_path, '../data')

    train_df = pd.read_csv(os.path.join(data_path, 'preprocessed_train.csv'), parse_dates=['datetime'],
                           index_col='datetime')
    val_df = pd.read_csv(os.path.join(data_path, 'preprocessed_val.csv'), parse_dates=['datetime'],
                         index_col='datetime')
    test_df = pd.read_csv(os.path.join(data_path, 'preprocessed_test.csv'), parse_dates=['datetime'],
                          index_col='datetime')

    return train_df, val_df, test_df


def prepare_prophet_df(df):
    df = df.reset_index()
    df['datetime'] = df['datetime'].dt.tz_localize(None)  # Remove timezone information
    df = df.rename(columns={'datetime': 'ds', 'carbon_intensity_avg': 'y'})
    return df[['ds', 'y']]


def tune_prophet(train_df, val_df):
    prophet_df = pd.concat([train_df, val_df])
    prophet_df = prepare_prophet_df(prophet_df)

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0]
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    best_params = None
    best_mse = float('inf')

    for params in all_params:
        model = Prophet(**params)
        model.fit(prophet_df)

        # Predict on validation data
        val_df_prepared = prepare_prophet_df(val_df)
        val_pred = model.predict(val_df_prepared[['ds']])

        mse = mean_squared_error(val_df_prepared['y'], val_pred['yhat'])

        if mse < best_mse:
            best_mse = mse
            best_params = params

    print(f'Best params: {best_params}')
    print(f'Validation MSE: {best_mse}')

    return best_params


def train_model(train_df, val_df, best_params):
    prophet_df = pd.concat([train_df, val_df])
    prophet_df = prepare_prophet_df(prophet_df)

    model = Prophet(**best_params)
    model.fit(prophet_df)

    # Predict on validation data
    val_df_prepared = prepare_prophet_df(val_df)
    val_pred = model.predict(val_df_prepared[['ds']])

    mse = mean_squared_error(val_df_prepared['y'], val_pred['yhat'])
    mae = mean_absolute_error(val_df_prepared['y'], val_pred['yhat'])
    print(f'Validation MSE: {mse}')
    print(f'Validation MAE: {mae}')

    return model


def evaluate_model(model, test_df):
    test_df_prepared = prepare_prophet_df(test_df)
    test_pred = model.predict(test_df_prepared[['ds']])

    if test_pred['yhat'].isnull().any():
        print("Warning: NaN values found in predictions. Replacing NaNs with zeros.")
        test_pred['yhat'] = test_pred['yhat'].fillna(0)

    if test_df_prepared['y'].isnull().any():
        print("Warning: NaN values found in test target data. Replacing NaNs with zeros.")
        test_df_prepared['y'] = test_df_prepared['y'].fillna(0)

    mse = mean_squared_error(test_df_prepared['y'], test_pred['yhat'])
    mae = mean_absolute_error(test_df_prepared['y'], test_pred['yhat'])
    print(f'Test MSE: {mse}')
    print(f'Test MAE: {mae}')


if __name__ == "__main__":
    train_df, val_df, test_df = load_data()
    best_params = tune_prophet(train_df, val_df)
    model = train_model(train_df, val_df, best_params)
    evaluate_model(model, test_df)

