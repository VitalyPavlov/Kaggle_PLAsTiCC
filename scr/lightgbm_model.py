import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from hyperopt import tpe, hp, fmin
import time
from datetime import datetime as dt


SEED = 1  # random_seed
N_FOLD = 5  # number of folds in StratifiedKFold
MAX_CALLS = 20  # number of iterations for hyperparameter tuning
CHUNKS = 100000
CLASS_WEIGHTS_GAL = {6: 1, 16: 1, 53: 1, 65: 1, 92: 1}
CLASS_WEIGHTS_EXTRA = {15: 2, 42: 1, 52: 1, 62: 1, 64: 2, 67: 1, 88: 1, 90: 1, 95: 1}
TUNING = 'skopt' # 'skopt' or 'hyperopt'

# Tuning LightGBM parameters
space_skopt = [Integer(4, 7, name='max_depth'),
               Real(low=1e-3, high=1e-1, prior="log-uniform", name='learning_rate'),
               Integer(low=100, high=800, name='n_estimators')]

space_hyperopt = {'max_depth': hp.choice('max_depth', range(4, 8, 1)),
                  'learning_rate': hp.loguniform('learning_rate', np.log(1e-3), np.log(1e-1)),
                  'n_estimators': hp.choice('n_estimators', range(100, 801, 1))}


def multi_weighted_logloss(y_true, y_preds):
    """ Multi logloss for PLAsTiCC challenge """

    if len(np.unique(y_true)) == 5:
        class_weights = CLASS_WEIGHTS_GAL
    elif len(np.unique(y_true)) == 9:
        class_weights = CLASS_WEIGHTS_EXTRA
    else:
        class_weights = {**CLASS_WEIGHTS_GAL, **CLASS_WEIGHTS_EXTRA}

    y_p = y_preds.reshape(y_true.shape[0], -1, order='F')
    y_ohe = pd.get_dummies(y_true)

    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)

    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)

    # Weight average and divide by the number of positives
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)

    return 'wloss', loss, False


def smote_adataset(x_train, y_train, x_test, y_test):
    """ Oversampling """

    sm = SMOTE(random_state=SEED)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

    return x_train_res, pd.Series(y_train_res), x_test, pd.Series(y_test)


def lgbm_model(max_depth, learning_rate, n_estimators):
    """ LightGBM model """

    best_params.update({'max_depth': max_depth,
                        'learning_rate': learning_rate,
                        'n_estimators': n_estimators})

    # Compute weights
    w = y.value_counts()
    weights = {i: np.sum(w) / w[i] for i in w.index}

    clfs = []
    folds = StratifiedKFold(n_splits=N_FOLD,
                            shuffle=True,
                            random_state=SEED)

    oof_preds = np.zeros((len(full_train), np.unique(y).shape[0]))
    for fold_, (trn_, val_) in enumerate(folds.split(y, y)):
        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        trn_xa, trn_y, val_xa, val_y = smote_adataset(trn_x.values, trn_y.values, val_x.values, val_y.values)
        trn_x = pd.DataFrame(data=trn_xa, columns=trn_x.columns)
        val_x = pd.DataFrame(data=val_xa, columns=val_x.columns)

        clf = LGBMClassifier(**best_params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=multi_weighted_logloss,
            verbose=False,
            early_stopping_rounds=50,
            sample_weight=trn_y.map(weights)
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        _, loss, _ = multi_weighted_logloss(val_y, oof_preds[val_, :])
        print('no {}-fold loss: {}'.format(fold_ + 1, loss))

    _, loss, _ = multi_weighted_logloss(y_true=y, y_preds=oof_preds)
    print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(loss))

    return clfs, loss


@use_named_args(space_skopt)
def objective_skopt(max_depth, learning_rate, n_estimators):
    # Print the hyper-parameters.
    print('max depth:', max_depth)
    print('learning rate: {0:.2e}'.format(learning_rate))
    print('n_estimators:', n_estimators)

    _clfs, loss = lgbm_model(max_depth, learning_rate, n_estimators)
    return loss


def objective_hyperopt(args):
    # Print the hyper-parameters.
    print('max depth:', args['max_depth'])
    print('learning rate: {0:.2e}'.format(args['learning_rate']))
    print('n_estimators:', args['n_estimators'])

    _clfs, loss = lgbm_model(args['max_depth'], args['learning_rate'], args['n_estimators'])
    return loss


def save_importances(clfs, type_obj):
    """ Feature importance LightGBM """

    importances = pd.DataFrame()
    for fold_, clf in enumerate(clfs):
        imp_df = pd.DataFrame({
            'feature': full_train.columns,
            'gain': clf.feature_importances_,
            'fold': [fold_ + 1] * len(full_train.columns),
        })
        importances = pd.concat([importances, imp_df], axis=0)

    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    mean_gain.sort_values('gain', ascending=False, inplace=True)
    mean_gain.to_csv('../data/lgbm_model/lgbm_importances_{}_obj.csv'.format(type_obj), index=False)
    print(mean_gain.head(10))


def hyperparam_tuning(type_obj=None):
    """ Hyperparameters optimization using skopt or hyperopt"""

    if TUNING == 'skopt':
        best_lgbm = gp_minimize(func=objective_skopt,
                                dimensions=space_skopt,
                                acq_func='EI',
                                n_calls=MAX_CALLS,
                                random_state=SEED)

        max_depth = int(best_lgbm.x[0])
        learning_rate = float(best_lgbm.x[1])
        n_estimators = int(best_lgbm.x[2])

        best_params.update({'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'n_estimators': n_estimators})

    elif TUNING == 'hyperopt':
        best_lgbm = fmin(objective_hyperopt, space_hyperopt,
                         algo=tpe.suggest, max_evals=MAX_CALLS)

        max_depth = int(best_lgbm['max_depth'])
        learning_rate = float(best_lgbm['learning_rate'])
        n_estimators = int(best_lgbm['n_estimators'])

        best_params.update({'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'n_estimators': n_estimators})

    with open('../data/lgbm_model/lgbm_params_{}_obj.json'.format(type_obj), 'w') as f:
        json.dump(best_params, f, indent=4)

    return best_params


def fit(is_tuning, type_obj):
    """ Tuning and training LightGBM model """

    global full_train, y, best_params

    # Load train dataset
    full_train = pd.read_csv('../data/train_data.csv', compression='gzip')
    full_train = full_train.round(5)

    if type_obj == 'galactic':
        full_train = full_train[full_train.hostgal_photoz == 0].reset_index(drop=True)
    elif type_obj == 'extragalactic':
        full_train = full_train[full_train.hostgal_photoz != 0].reset_index(drop=True)

    y = full_train['target']
    del full_train['object_id'], full_train['target']

    with open('../data/lgbm_model/lgbm_params_{}_obj.json'.format(type_obj), 'r') as f:
        best_params = json.load(f)

    if is_tuning:
        best_params = hyperparam_tuning(type_obj=type_obj)

    max_depth = best_params['max_depth']
    learning_rate = best_params['learning_rate']
    n_estimators = best_params['n_estimators']

    clfs, loss = lgbm_model(max_depth=max_depth,
                            learning_rate=learning_rate,
                            n_estimators=n_estimators)

    # save feature importance
    save_importances(clfs, type_obj)

    return clfs, loss


def predict_chunk(df, clfs):
    """ Prediction for chunk """

    object_id = df['object_id'].values
    del df['object_id']

    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(df, num_iteration=clf.best_iteration_)
        else:
            preds += clf.predict_proba(df, num_iteration=clf.best_iteration_)

    preds = preds / len(clfs)

    # Create DataFrame from predictions
    preds_df = pd.DataFrame(preds,
                            columns=['class_{}'.format(s) for s in clfs[0].classes_])
    preds_df['object_id'] = object_id

    return preds_df


def fit_predict():
    """ Training and test prediction """

    # Fitting model for galactic and extragalactic objects
    clfs_gal, loss_gal = fit(is_tuning=False, type_obj='galactic')
    clfs_extra, loss_extra = fit(is_tuning=False, type_obj='extragalactic')

    # filename to save result for test dataset
    filename = 'result_lgbm_{:.4f}_{:.4f}_{}.csv'.format(loss_gal, loss_extra,
                                                         dt.now().strftime('%Y-%m-%d-%H-%M'))

    start = time.time()
    for ind, full_test_ in enumerate(pd.read_csv('../data/test_data.csv',
                                                 chunksize=CHUNKS,
                                                 iterator=True,
                                                 compression='gzip')):

        full_test_ = full_test_.round(5)
        full_test_gal = full_test_[full_test_.hostgal_photoz == 0].reset_index(drop=True)
        full_test_extra = full_test_[full_test_.hostgal_photoz != 0].reset_index(drop=True)

        preds_gal = predict_chunk(df=full_test_gal, clfs=clfs_gal)
        preds_extra = predict_chunk(df=full_test_extra, clfs=clfs_extra)

        if ind == 0:
            preds_gal.to_csv('../data/gal_' + filename, header=True, mode='a', index=False)
            preds_extra.to_csv('../data/extra_' + filename, header=True, mode='a', index=False)
        else:
            preds_gal.to_csv('../data/gal_' + filename, header=False, mode='a', index=False)
            preds_extra.to_csv('../data/extra_' + filename, header=False, mode='a', index=False)

        print('{:15d} done in {:5.1f} minutes'.format(
            CHUNKS * (ind + 1), (time.time() - start) / 60), flush=True)

    return filename


def main():
    is_predict = False  # True for test prediction else False

    if is_predict:
        filename = fit_predict()

        preds_gal = pd.read_csv('../data/gal_' + filename)
        preds_extra = pd.read_csv('../data/extra_' + filename)

        preds_gal = preds_gal.groupby('object_id').mean()
        preds_extra = preds_extra.groupby('object_id').mean()

        preds = pd.concat([preds_gal, preds_extra], axis=1)
        preds = preds.fillna(0)

        # Prediction class_99
        preds_99 = np.ones(preds.shape[0])
        for col in preds.columns:
            preds_99 *= (1 - preds[col])
        preds['class_99'] = 0.18 * preds_99 / np.mean(preds_99)

        # Saving result
        preds = preds[sorted(preds.columns, key=lambda x: int(x.split('_')[-1]))]
        preds.to_csv('../data/sub_' + filename)
        print('RESULT IS SAVED IN sub_{}'.format(filename))
    else:
        type_obj = 'galactic'  # 'galactic' or 'extragalactic'
        is_tuning = False
        _clfs, loss = fit(is_tuning=is_tuning, type_obj=type_obj)
        print('MULTI WEIGHTED LOG LOSS: {:.5f}'.format(loss))


if __name__ == '__main__':
    main()
