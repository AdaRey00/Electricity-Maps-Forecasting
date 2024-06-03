import argparse
import os
import pandas as pd
from preprocessing import preprocess_data
from models.model import tune_prophet, train_model, evaluate_model


def main(data_path):
    # Check if preprocessed files exist
    if not (os.path.exists('data/preprocessed_train.csv') and
            os.path.exists('data/preprocessed_val.csv') and
            os.path.exists('data/preprocessed_test.csv')):
        preprocess_data(data_path)

    # Load preprocessed data
    train_df = pd.read_csv('data/preprocessed_train.csv', parse_dates=['datetime'], index_col='datetime')
    val_df = pd.read_csv('data/preprocessed_val.csv', parse_dates=['datetime'], index_col='datetime')
    test_df = pd.read_csv('data/preprocessed_test.csv', parse_dates=['datetime'], index_col='datetime')

    # Tune model parameters
    best_params = tune_prophet(train_df, val_df)

    # Train and evaluate model with the best parameters
    model = train_model(train_df, val_df, best_params)
    evaluate_model(model, test_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/DK-DK2.csv', help='Path to the dataset')
    args = parser.parse_args()
    main(args.data_path)
