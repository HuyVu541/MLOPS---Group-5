import pandas as pd
import numpy as np

def preprocess_data(df):
    df_numerical = df.select_dtypes(include = 'number')
    df_numerical['time'] = df['time']
    df = df_numerical
    df['match_match_price'] = df['match_match_price'].replace(0, np.nan)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    df.dropna(axis = 1)
    return df  