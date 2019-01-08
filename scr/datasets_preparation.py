import os
import pandas as pd
import feature_extraction
import time


BASE_DIR = os.path.dirname("~/Documents/PythonScripts/Kaggle/PLASTICC/")
CHUNKS = 5000000

# agg features
aggs = {
    'flux': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'flux_err': ['min', 'max', 'mean', 'median', 'std', 'skew'],
    'detected': ['mean'],
    'flux_ratio_sq': ['sum', 'skew'],
    'flux_by_flux_ratio_sq': ['sum', 'skew']
}

# tsfresh features
fcp = {
    'flux': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None,
        'mean_change': None,
        'mean_abs_change': None,
        'length': None,
        'cid_ce': [{'normalize': True}, {'normalize': False}]
    },

    'flux_by_flux_ratio_sq': {
        'longest_strike_above_mean': None,
        'longest_strike_below_mean': None
    },

    'flux_passband': {
        'fft_coefficient': [
            {'coeff': 0, 'attr': 'abs'},
            {'coeff': 1, 'attr': 'abs'}
        ],
        'kurtosis': None,
        'skewness': None,
        'mean': None
    },

    'mjd': {
        'maximum': None,
        'minimum': None,
        'mean_change': None,
        'mean_abs_change': None
    },

    'difference': {
        'sum_values': None
    }
}

def main():
    # Train data preparation
    meta_train = feature_extraction.process_meta(os.path.join(BASE_DIR, 'training_set_metadata.csv'))
    train = pd.read_csv(os.path.join(BASE_DIR, 'training_set.csv'))
    full_train = feature_extraction.featurize(train, meta_train, aggs, fcp, n_jobs=3)

    train_mean = full_train.mean(axis=0)
    full_train.fillna(train_mean, inplace=True)
    full_train.to_csv('../data/train_data.csv', index=False, compression='gzip')

    # Test data preparation
    meta_test = feature_extraction.process_meta(os.path.join(BASE_DIR, 'test_set_metadata.csv'))
    start = time.time()
    for ind, test_ in enumerate(pd.read_csv(os.path.join(BASE_DIR, 'test_set.csv.zip'), chunksize=CHUNKS, iterator=True)):
        full_test = feature_extraction.featurize(test_, meta_test, aggs, fcp, n_jobs=3)
        full_test.fillna(train_mean, inplace=True)
        if ind == 0:
            full_test.to_csv('../data/test_data.csv', header=True, mode='a', index=False, compression='gzip')
        else:
            full_test.to_csv('../data/test_data.csv', header=False, mode='a', index=False, compression='gzip')

        if (ind + 1) % 10 == 0:
            print('%15d done in %5.1f' % (CHUNKS * (ind + 1), (time.time() - start) / 60))


if __name__ == '__main__':
    main()
