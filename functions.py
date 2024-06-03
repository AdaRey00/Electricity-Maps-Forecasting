import pandas as pd
def apply_rolling_median(df, features, window_size):
    smoothed_data = {}
    for feature in features:
        smoothed_data[f'smoothed_{feature}'] = df[feature].rolling(window=window_size, center=True).median()
    smoothed_df = pd.DataFrame(smoothed_data)
    return smoothed_df