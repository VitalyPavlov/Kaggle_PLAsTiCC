import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from collections import Counter
import time
from datetime import datetime as dt
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras import optimizers


SEED = 1  # random_seed
N_FOLD = 5  # number of folds in StratifiedKFold
EPOCHS = 600
BATCH_SIZE = 100
CHUNKS = 100000
CLASS_MAP = {6: 0, 15: 1, 16: 2, 42: 3, 52: 4, 53: 5, 62: 6,
             64: 7, 65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13}
CLASS_WEIGHTS = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1,
                 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
WTABLE = [0.01924057, 0.06307339, 0.117737, 0.15201325, 0.02331804, 0.00382263,
          0.06167176, 0.01299694, 0.125, 0.02650357, 0.04714577, 0.29472477,
          0.03045362, 0.02229867]
BASE_DIR = os.path.dirname("~/Documents/PythonScripts/Kaggle/PLASTICC/")


def multi_weighted_logloss(y_true, y_preds):
    """ Multi logloss for PLAsTiCC challenge """

    y_p = np.clip(a=y_preds, a_min=1e-15, a_max=1 - 1e-15)
    y_log_ones = np.sum(y_true * np.log(y_p), axis=0)

    nb_pos = y_true.sum(axis=0).astype(float)
    class_arr = np.array([CLASS_WEIGHTS[k] for k in sorted(CLASS_WEIGHTS.keys())])
    y_w = y_log_ones * class_arr / nb_pos
    loss = - np.sum(y_w) / np.sum(class_arr)

    return loss


def wloss_metrics(y_true, y_pred):
    """ Loss function for MLP """

    weight_tensor = tf.convert_to_tensor(list(CLASS_WEIGHTS.values()), dtype=tf.float32)
    y_h = tf.convert_to_tensor(y_true)
    y_h /= tf.reduce_sum(y_h, axis=0, keepdims=True) + 1
    y_p = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    ln_p = tf.log(tf.clip_by_value(y_p, 1e-15, 1-1e-15))
    wll = tf.reduce_sum(y_h * ln_p, axis=0)
    loss = -tf.reduce_sum(weight_tensor * wll) / tf.reduce_sum(weight_tensor)
    return loss


def wloss(y_true, y_pred):
    """ Loss function for MLP """

    yc = tf.clip_by_value(y_pred, 1e-15, 1 - 1e-15)
    loss = -(tf.reduce_mean(tf.reduce_mean(y_true * tf.log(yc), axis=0) / WTABLE))
    return loss


def nn_model(input_dim, dropout_rate=0.5, activation='relu'):
    """ Building MLP model """
    start_neurons = 512

    model = Sequential()
    model.add(Dense(start_neurons, input_dim=input_dim, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 2, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 4, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(Dense(start_neurons // 8, activation=activation))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate / 2))

    model.add(Dense(len(CLASS_MAP), activation='softmax'))
    return model


def fit():
    """ Training MLP model """

    # Load train dataset
    full_train = pd.read_csv('../data/train_data.csv', compression='gzip')

    y = full_train['target']
    del full_train['object_id'], full_train['target']

    features = full_train.columns

    ss = StandardScaler()
    full_train = ss.fit_transform(full_train)

    y_map = np.array([CLASS_MAP[val] for val in y])
    y_categorical = to_categorical(y_map)

    clfs = []
    folds = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    oof_preds = np.zeros((len(full_train), len(CLASS_MAP)))

    K.clear_session()
    check_point = ModelCheckpoint('../data/nn_model/keras.model', monitor='val_wloss_metrics',
                                  mode='min', save_best_only=True, verbose=0)

    for fold_, (trn_, val_) in enumerate(folds.split(y_map, y_map)):
        x_train, y_train = full_train[trn_], y_categorical[trn_]
        x_valid, y_valid = full_train[val_], y_categorical[val_]

        model = nn_model(input_dim=full_train.shape[1], dropout_rate=0.5, activation='tanh')
        model.compile(loss=wloss, optimizer=optimizers.Adam(lr=0.001), metrics=[wloss_metrics])
        model.fit(x_train, y_train,
                  validation_data=[x_valid, y_valid],
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  shuffle=True,
                  verbose=2,
                  callbacks=[check_point])

        print('Loading Best Model')
        model.load_weights('../data/nn_model/keras.model')
        oof_preds[val_, :] = model.predict_proba(x_valid, batch_size=BATCH_SIZE)
        print('MULTI WEIGHTED LOG LOSS : %.5f ' %
              multi_weighted_logloss(y_valid, model.predict_proba(x_valid, batch_size=BATCH_SIZE)))
        clfs.append(model)

    loss = multi_weighted_logloss(y_categorical, oof_preds)
    print('MULTI WEIGHTED LOG LOSS : %.5f ' % loss)

    # filename to save result for test dataset
    filename = 'result_nn_{:4f}_{}.csv'.format(loss, dt.now().strftime('%Y-%m-%d-%H-%M'))

    return clfs, ss, filename, features


def predict_chunk(df, clfs, ss, features):
    """ Prediction for chunk """

    object_id = df['object_id'].values
    del df['object_id']

    df = ss.transform(df[features])

    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(df)
        else:
            preds += clf.predict_proba(df)

    preds = preds / len(clfs)

    # Create DataFrame from predictions
    preds_df = pd.DataFrame(preds,
                            columns=['class_{}'.format(s) for s in sorted(CLASS_MAP.keys())])
    preds_df['object_id'] = object_id

    return preds_df


def predict(clfs, ss, filename, features):
    """ Test prediction """

    start = time.time()
    for ind, full_test_ in enumerate(pd.read_csv('../data/test_data.csv',
                                                 chunksize=CHUNKS,
                                                 iterator=True,
                                                 compression='gzip')):

        preds_ = predict_chunk(full_test_, clfs, ss, features)

        if ind == 0:
            preds_.to_csv('../data/nn_model/' + filename, header=True, mode='a', index=False)
        else:
            preds_.to_csv('../data/nn_model/' + filename, header=False, mode='a', index=False)

        print('{:15d} done in {:5.1f} minutes'.format(
            CHUNKS * (ind + 1), (time.time() - start) / 60), flush=True)


def main():
        clfs, ss, filename, features = fit()

        predict(clfs, ss, filename, features)

        res = pd.read_csv('../data/nn_model/' + filename)
        res = res.groupby('object_id').mean()

        preds_99 = np.ones(res.shape[0])
        for col in res.columns:
            preds_99 *= (1 - res[col])

        res['class_99'] = 0.18 * preds_99 / np.mean(preds_99)

        meta_test = pd.read_csv(os.path.join(BASE_DIR, 'test_set_metadata.csv'))
        meta_test.set_index('object_id', inplace=True)

        gal_obj = ['class_6', 'class_16', 'class_53', 'class_65', 'class_92']
        res.loc[meta_test[meta_test.hostgal_photoz != 0].index, gal_obj] = 0

        extra_obj = ['class_15', 'class_42', 'class_52', 'class_62',
                     'class_64', 'class_67', 'class_88', 'class_90', 'class_95']
        res.loc[meta_test[meta_test.hostgal_photoz == 0].index, extra_obj] = 0

        res.to_csv('../data/nn_model/sub_{}'.format(filename), index=True)


if __name__ == "__main__":
    main()
