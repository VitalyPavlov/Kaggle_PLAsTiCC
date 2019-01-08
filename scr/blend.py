import pandas as pd


KOEF = 0.7


def main():
    filename_lgbm = 'sub_result_lgbm_0.1108_0.7888_2019-01-01-12-30.csv'
    filename_nn = 'sub_result_nn_0.619605_2019-01-08-19-24.csv'

    lgbm_df = pd.read_csv('../data/lgbm_model/' + filename_lgbm)
    nn_df = pd.read_csv('../data/nn_model/' + filename_nn)

    lgbm_df.set_index('object_id', inplace=True)
    nn_df.set_index('object_id', inplace=True)

    blend_df = KOEF * lgbm_df + (1 - KOEF) * nn_df
    blend_df.to_csv('../data/sub_blend_result_0.1108_0.7888_0.619605.csv')


if __name__ == '__main__':
    main()
