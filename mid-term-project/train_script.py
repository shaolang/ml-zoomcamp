from joblib import dump
import numpy as np
import pandas as pd
import train

if __name__ == '__main__':
    df = pd.read_csv('resale-flat-price-2017-2021.csv')
    trainX, trainy, _testX, _testy = train.split_train_test(df)
    pipeline = train.create_pipeline()
    pipeline.fit(trainX, np.log1p(trainy))

    dump(pipeline, 'pipeline.bin')
