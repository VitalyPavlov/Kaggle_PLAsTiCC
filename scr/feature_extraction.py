import pandas as pd
import numpy as np
from numba import jit
from tsfresh.feature_extraction import extract_features


@jit
def features_before_agg(df):
    """ Extracting features before aggregation """

    df['flux_difference'] = df.flux.diff(1)
    df['flux_difference'].fillna(0, inplace=True)

    flux_ratio_sq = np.power(df['flux'].values / df['flux_err'].values, 2.0)

    df_flux = pd.DataFrame({
        'flux_ratio_sq': flux_ratio_sq,
        'flux_by_flux_ratio_sq': df['flux'].values * flux_ratio_sq},
        index=df.index)

    return pd.concat([df, df_flux], axis=1)


@jit
def features_after_agg(df):
    """ Extracting features after aggregation """

    flux_w_mean = df['flux_by_flux_ratio_sq_sum'].values / df['flux_ratio_sq_sum'].values
    flux_diff = df['flux_max'].values - df['flux_min'].values

    df_flux_agg = pd.DataFrame({
        'flux_w_mean': flux_w_mean,
        'flux_diff1': flux_diff,
        'flux_diff2': flux_diff / df['flux_mean'].values,
        'flux_diff3': flux_diff / flux_w_mean,
    }, index=df.index)

    return pd.concat([df, df_flux_agg], axis=1)


def tsfreash_features(df, column_sort, column_kind, column_value, fc_parameters, n_jobs):
    """ Extracting aggregation features using tsfreash """

    agg_df = extract_features(df,
                              column_id='object_id',
                              column_sort=column_sort,
                              column_kind=column_kind,
                              column_value=column_value,
                              default_fc_parameters=fc_parameters,
                              n_jobs=n_jobs)

    if column_kind:
        agg_df.columns = [str(column_value) + '_' + c for c in agg_df.columns]
    agg_df.index.rename('object_id', inplace=True)

    return agg_df


@jit
def photoz2dist(df):
    """ Distmod as a function of hostgal_photoz """

    hostgal_photoz = df["hostgal_photoz"].values
    distmod = ((((((np.log((((hostgal_photoz) +
                          (np.log((((hostgal_photoz) +
                                    (np.sqrt((np.log((np.maximum(((3.0)),
                                                                 ((((hostgal_photoz) * 2.0))))))))))))))))) +
                ((12.99870681762695312)))) + ((1.17613816261291504)))) * (3.0))

    return distmod


def featurize(df, df_meta, aggs, fcp, n_jobs=2):
    """ Extracting features from train/test set """

    df = features_before_agg(df)

    # Extracting aggregation features for all dataset
    agg_df_all = df.groupby('object_id').agg(aggs)
    agg_df_all.columns = ['{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df_all = features_after_agg(agg_df_all)
    agg_df_all.columns = ['all_' + c for c in agg_df_all.columns]

    # Extracting aggregation features for datected = 1
    agg_df_detected = df[df.detected == 1].groupby('object_id').agg(aggs)
    agg_df_detected.columns = ['{}_{}'.format(k, agg) for k in aggs.keys() for agg in aggs[k]]
    agg_df_detected = features_after_agg(agg_df_detected)

    # Features obtained with tsfresh
    agg_df_flux_passband = tsfreash_features(df,
                                             column_sort='mjd',
                                             column_kind='passband',
                                             column_value='flux',
                                             fc_parameters=fcp['flux_passband'],
                                             n_jobs=n_jobs)


    agg_df_diff_passband = tsfreash_features(df,
                                             column_sort='mjd',
                                             column_kind='passband',
                                             column_value='flux_difference',
                                             fc_parameters=fcp['difference'],
                                             n_jobs=n_jobs)


    agg_df_flux = tsfreash_features(df,
                                    column_sort=None,
                                    column_kind=None,
                                    column_value='flux',
                                    fc_parameters=fcp['flux'],
                                    n_jobs=n_jobs)

    agg_df_flux_by_flux_ratio_sq = tsfreash_features(df,
                                                     column_sort=None,
                                                     column_kind=None,
                                                     column_value='flux_by_flux_ratio_sq',
                                                     fc_parameters=fcp['flux_by_flux_ratio_sq'],
                                                     n_jobs=n_jobs)

    agg_df_mjd = tsfreash_features(df[df['detected'] == 1],
                                   column_sort=None,
                                   column_kind=None,
                                   column_value='mjd',
                                   fc_parameters=fcp['mjd'],
                                   n_jobs=n_jobs)

    agg_df_mjd['mjd_diff_det'] = agg_df_mjd['mjd__maximum'].values - agg_df_mjd['mjd__minimum'].values
    del agg_df_mjd['mjd__maximum'], agg_df_mjd['mjd__minimum']

    # Concat all dataframes
    agg_df_ts = pd.concat([agg_df_all,
                           agg_df_detected,
                           agg_df_flux_passband,
                           agg_df_flux,
                           agg_df_flux_by_flux_ratio_sq,
                           agg_df_mjd,
                           agg_df_diff_passband
                           ], axis=1).reset_index()

    result = agg_df_ts.merge(right=df_meta, how='left', on='object_id')

    # Fill nan values of distmod
    is_galactic_obj = df_meta.hostgal_photoz == 0
    result.loc[result.distmod.isnull() & ~is_galactic_obj, 'distmod'] = \
        photoz2dist(result.loc[result.distmod.isnull() & ~is_galactic_obj])
    result.loc[result.distmod.isnull() & is_galactic_obj, 'distmod'] = 0

    # Del unused features
    del result['hostgal_specz']
    del result['ra'], result['decl'], result['gal_l'], result['gal_b']
    del result['ddf']

    return result


@jit
def haversine_plus(lon1, lat1, lon2, lat2):
    """ Haversine formula """

    # Convert decimal degrees to Radians:
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    # Implementing Haversine Formula:
    dlon = np.subtract(lon2, lon1)
    dlat = np.subtract(lat2, lat1)

    a = np.add(np.power(np.sin(np.divide(dlat, 2)), 2),
               np.multiply(np.cos(lat1),
                           np.multiply(np.cos(lat2),
                                       np.power(np.sin(np.divide(dlon, 2)), 2))))

    haversine = np.multiply(2, np.arcsin(np.sqrt(a)))

    out_dict = {
        'haversine': haversine,
        'latlon1': np.subtract(np.multiply(lon1, lat1), np.multiply(lon2, lat2)),
    }
    return out_dict


def process_meta(filename):
    """ Feature preparation of metafile """

    meta_df = pd.read_csv(filename)

    meta_dict = dict()
    meta_dict.update(haversine_plus(meta_df['ra'].values, meta_df['decl'].values,
                                    meta_df['gal_l'].values, meta_df['gal_b'].values))

    meta_dict['hostgal_photoz_certain'] = np.multiply(
        meta_df['hostgal_photoz'].values,
        np.exp(meta_df['hostgal_photoz_err'].values))

    meta_df = pd.concat([meta_df, pd.DataFrame(meta_dict, index=meta_df.index)], axis=1)

    return meta_df
