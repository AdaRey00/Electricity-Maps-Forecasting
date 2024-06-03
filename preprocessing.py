import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from features import selected_features
from functions import apply_rolling_median
from itertools import combinations

def preprocess_data(file_path):
    # load the dataset
    df = pd.read_csv(file_path)
    print("Initial shape:", df.shape)

    # drop columns with all missing values
    df.dropna(axis=1, how='all', inplace=True)
    print("After dropping columns with all missing values:", df.shape)

    # drop rows with more than a certain threshold of missing values
    df.dropna(axis=0, thresh=57, inplace=True)
    print("After dropping rows with more than a threshold of missing values:", df.shape)

    # drop unnecessary columns
    df.drop(['production_sources', 'timestamp'], axis=1, inplace=True)
    print("After dropping unnecessary columns:", df.shape)

    # handle the 'zone_name' column with one-hot encoding
    unique_zones = df['zone_name'].unique()
    if len(unique_zones) > 1:
        df = pd.get_dummies(df, columns=['zone_name'], prefix='zone')
    else:
        df['zone_name'] = 1
    print("After handling 'zone_name' column:", df.shape)

    # convert 'datetime' column to datetime type
    df['datetime'] = pd.to_datetime(df['datetime'])
    print("Datetime conversion done.")

    # sort the dataframe by datetime to ensure the order is correct
    df.sort_values('datetime', inplace=True)

    # extract the time features from datetime column but do not drop the column
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    print("After extracting datetime features:", df.shape)

    # transform cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df.drop(columns=['hour', 'dayofweek'], inplace=True)
    print("After transforming cyclical features:", df.shape)

    # retain only the selected features
    retained_features = selected_features + ['carbon_intensity_avg']
    retained_features = [col for col in retained_features if col in df.columns]
    df = df[retained_features + ['datetime']]
    print("After retaining selected features:", df.shape)

    # convert object columns to appropriate types before interpolation
    df = df.infer_objects()
    print("Missing values before KNN imputation:", df.isnull().sum().sum())

    # initialize KNNImputer and apply it only to numeric columns
    imputer = KNNImputer(n_neighbors=5)
    numeric_features = [col for col in retained_features if col != 'datetime']

    # apply KNN imputation
    df[numeric_features] = imputer.fit_transform(df[numeric_features])
    print("Missing values after KNN imputation:", df.isnull().sum().sum())

    # set window size
    window_size = 5

    # apply rolling median smoothing
    apply_rolling_median(df, numeric_features, window_size)

    # add lag features
    lag_features = ['carbon_intensity_avg']
    for feature in lag_features:
        for lag in range(1, 25):
            df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)

    # drop rows with NaN values generated by rolling statistics and lag features
    df.dropna(inplace=True)
    print("After dropping rows with NaN values from rolling statistics and lag features:", df.shape)

    # creation of interaction features
    interaction_features = [
        'power_origin_percent_fossil_avg',
        'carbon_rate_avg',
        'power_consumption_coal_avg',
        'power_origin_percent_renewable_avg'
    ]
    for feature1, feature2 in combinations(interaction_features, 2):
        interaction_term_name = f'{feature1}_x_{feature2}'
        df[interaction_term_name] = df[feature1] * df[feature2]

    print("After creating interaction features:", df.shape)

    # define the preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # apply the preprocessing pipeline to all features except the target and datetime
    all_features_except_target_and_datetime = df.columns.difference(['carbon_intensity_avg', 'datetime'])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, all_features_except_target_and_datetime)
        ]
    )

    # apply the preprocessing pipeline
    df[all_features_except_target_and_datetime] = preprocessor.fit_transform(df[all_features_except_target_and_datetime])
    print("After applying preprocessing pipeline:", df.shape)

    # calculate the index for splitting
    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.15)  # 15% for validation
    test_size = len(df) - train_size - val_size  # remaining 15% for test

    # split the data into training, validation, and test sets
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]

    # save preprocessed data for modeling
    train_df.to_csv('data/preprocessed_train.csv', index=True)
    val_df.to_csv('data/preprocessed_val.csv', index=True)
    test_df.to_csv('data/preprocessed_test.csv', index=True)

    print("Preprocessing completed and data saved.")

# execute the preprocessing function
preprocess_data("data/DK-DK2.csv")
