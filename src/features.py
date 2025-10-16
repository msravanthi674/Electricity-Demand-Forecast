"""
Feature engineering utilities: cyclical encoding, lags, rolling stats.
"""

import numpy as np
import pandas as pd # type: ignore

def add_time_features(df):
    df = df.copy()
    # index is timezone-aware DatetimeIndex
    df['hour'] = df.index.hour
    df['dow'] = df.index.dayofweek  # Monday=0
    # cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # day-of-week one-hot (optional; keep as ints)
    dow_dummies = pd.get_dummies(df['dow'], prefix='dow')
    df = pd.concat([df, dow_dummies], axis=1)
    return df

def make_lag_features(df, lags=[1,2,3]):
    df = df.copy()
    for l in lags:
        df[f"lag_{l}"] = df['hourly_kwh'].shift(l)
    df['rolling24_mean'] = df['hourly_kwh'].rolling(window=24, min_periods=1).mean().shift(1)
    return df

def build_features(df, include_temp=False):
    df2 = add_time_features(df)
    df2 = make_lag_features(df2, lags=[1,2,3])
    # select features
    feats = ['hour_sin','hour_cos','rolling24_mean','lag_1','lag_2','lag_3']
    dow_cols = [c for c in df2.columns if c.startswith("dow_")]
    feats += dow_cols
    if include_temp and 'temperature_2m' in df2.columns:
        df2['temp'] = df2['temperature_2m']
        feats.append('temp')
    X = df2[feats].copy()
    return X, df2
