from datetime import datetime
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import numpy as np
import pandas as pd
import re

def split_train_test(df):
    train, test = train_test_split(df, test_size=0.2, random_state=42,
            stratify=df[['town']])

    return (train.drop('resale_price', axis=1), train['resale_price'],
            test.drop('resale_price', axis=1), test['resale_price'])

def convert_remaining_lease_to_years(df):
    pattern = re.compile(r'^(\d+)\D+(\d+)?')

    def convert(s):
        m = pattern.match(s)
        years = int(m[1], base=10)

        if m.lastindex == 1:
            return years
        else:
            return round(years + float(m[2]) / 12, 1)

    df.loc[:, 'remaining_lease'] = df.loc[:, 'remaining_lease'].apply(convert)

    return df


def convert_month_to_ordinal(df):
    df['month'] = df['month'].apply(
            lambda s: datetime.strptime(s, '%Y-%m').date().toordinal())

    return df


def create_pipeline():
    return Pipeline([
        ('remaining-lease-converter',
            FunctionTransformer(convert_remaining_lease_to_years)),
        ('month-converter', FunctionTransformer(convert_month_to_ordinal)),
        ('column-transformer',
            ColumnTransformer([
                ('scaler', StandardScaler(),
                    ['month', 'lease_commence_date', 'remaining_lease',
                        'floor_area_sqm']),
                ('one-hot', OneHotEncoder(),
                    ['town', 'flat_type', 'storey_range', 'flat_model'])
                ])),
        ('random-forest-regressor',
            RandomForestRegressor(max_depth=36, n_estimators=30, n_jobs=-1))
    ])
