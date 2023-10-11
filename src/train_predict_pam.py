# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Module to train and prediction using XGBoost Classifier
'''
# !/usr/bin/env python
# coding: utf-8
# pylint: disable=import-error
import time
from datetime import datetime
import warnings
import argparse
import sys
import logging
import numpy as np
import xgboost as xgb

if __name__ == "__main__":
    # Data size
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--file',
                        type=str,
                        required=False,
                        default='data_25000.pkl',
                        help='input pkl file name')
    parser.add_argument('-p',
                        '--package',
                        type=str,
                        required=False,
                        default='pandas',
                        help='data package to be used (pandas, modin)')
    parser.add_argument('-t',
                        '--tuning',
                        type=str,
                        required=False,
                        default='0',
                        help='hyper parameter tuning (0/1)')
    parser.add_argument('-cv',
                        '--cross-validation',
                        type=int,
                        required=False,
                        default=2,
                        help='cross validation iteration')
    parser.add_argument('-ncpu',
                        '--num-cpu',
                        type=int,
                        required=True,
                        default=4,
                        help='number of cpu cores, default 4')
    parser.add_argument('-d',
                    '--debug',
                    action='store_true',
                    help='changes logging level from INFO to DEBUG')

    FLAGS = parser.parse_args()

    if FLAGS.debug:
        logging_level=logging.DEBUG
    else:
        logging_level=logging.INFO

    logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)
    warnings.filterwarnings("ignore")

    pkg = FLAGS.package
    cv_val = FLAGS.cross_validation
    TUNING = False
    if FLAGS.tuning == "1":
        TUNING = True

    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler

    if pkg == "pandas":
        import pandas as pd   # noqa: F811

    if pkg == "modin":
        import modin.config as cfg
        cfg.Engine.put('ray')
        import modin.pandas as pd   # noqa: F811

    # Generating our data
    logger.info('Reading the dataset from %s...', FLAGS.file)
    try:
        data = pd.read_pickle(FLAGS.file)
    except FileNotFoundError:
        sys.exit('Dataset file not found')

    datasize = data.shape

    start = time.time()
    X = data.drop('Asset_Label', axis=1)
    y = data.Asset_Label

    # original split .25
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

    df_num_train = X_train.select_dtypes(['float', 'int', 'int32'])
    df_num_test = X_test.select_dtypes(['float', 'int', 'int32'])
    robust_scaler = RobustScaler()
    X_train_scaled = robust_scaler.fit_transform(df_num_train)
    X_test_scaled = robust_scaler.transform(df_num_test)

    # Making them pandas dataframes
    X_train_scaled_transformed = pd.DataFrame(X_train_scaled,
                                              index=df_num_train.index,
                                              columns=df_num_train.columns)
    X_test_scaled_transformed = pd.DataFrame(X_test_scaled,
                                             index=df_num_test.index,
                                             columns=df_num_test.columns)

    del X_train_scaled_transformed['Number_Repairs']
    del X_train_scaled_transformed['Tele_Attached']

    del X_test_scaled_transformed['Number_Repairs']
    del X_test_scaled_transformed['Tele_Attached']

    # Dropping the unscaled numerical columns
    X_train = X_train.drop(['Age', 'Elevation', 'Pole_Height', 'Measured_Length'], axis=1)
    X_test = X_test.drop(['Age', 'Elevation', 'Pole_Height', 'Measured_Length'], axis=1)

    # Creating train and test data with scaled numerical columns
    X_train_scaled_transformed = pd.concat([X_train_scaled_transformed, X_train], axis=1)
    X_test_scaled_transformed = pd.concat([X_test_scaled_transformed, X_test], axis=1)

    def fit_xgb_model(x_data, y_data):
        """Use a XGBClassifier for this problem."""
        # prepare data for xgboost training
        dtrain = xgb.DMatrix(x_data, y_data, nthread=FLAGS.num_cpu)
        label = dtrain.get_label()
        ratio = float(np.sum(label == 0)) / np.sum(label == 1)
        # Set xgboost parameters
        parameters = {'scale_pos_weight': ratio.round(2), 'n_jobs': FLAGS.num_cpu, 'tree_method': 'hist'}

        # define the model to use
        if TUNING is False:
            xg_cl = xgb.XGBClassifier(use_label_encoder=False)
        else:
            xg_cl = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        xg_cl.set_params(**parameters)
        # Train the model
        xg_cl.fit(x_data, y_data)
        return xg_cl

    X_train_scaled_transformed = X_train_scaled_transformed.astype(
                                    {'Tele_Attached': 'float64',
                                     'Number_Repairs': 'float64'})
    X_test_scaled_transformed = X_test_scaled_transformed.astype(
                                    {'Tele_Attached': 'float64',
                                     'Number_Repairs': 'float64'})

    # Training
    tstart = time.time()
    xgb_model = fit_xgb_model(X_train_scaled_transformed, y_train)
    ttime = time.time() - tstart
    
    if TUNING is True:
        logger.info("Starting hyper parameter tuning:")
        # GridSearchCV
        def timer(start_time=None):  # pylint: disable=missing-function-docstring
            if not start_time:
                start_time = datetime.now()
                return start_time
            if start_time:
                thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
                tmin, tsec = divmod(temp_sec, 60)
                print('Time taken: %i hours %i minutes and %s seconds.',
                      (thour, tmin, round(tsec, 2)))
            return 0

        # Hyper parameters for tuning
        # n_jobs should be used according to the underlying HW accelerators, hence -1 is given
        params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            # 'subsample': [0.6, 0.8, 1.0],
            # 'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'n_jobs': [-1]
            # 'learning_rate': [0.001, 0.01]
            }

        xgb_model = GridSearchCV(xgb_model, param_grid=params, cv=cv_val, verbose=10)
        xgb_model.fit(X_train_scaled_transformed, y_train)

    pstart = time.time()
    xgb_prediction = xgb_model.predict(X_test_scaled_transformed)
    ptime = time.time() - pstart
    xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(y_test))

    xgb_total = ptime

    y_test = np.ravel(y_test)

    etime = time.time() - start
    accuracy_scr = 1 - xgb_errors_count / xgb_prediction.shape[0]
    sys.stdout.flush()
    sys.stderr.flush()
    logger.info('=====> Total Time: %f secs for data size %s', etime, datasize)
    logger.info('=====> Training Time %f secs', ttime)
    logger.info('=====> Prediction Time %f secs', ptime)
    logger.info('=====> XGBoost accuracy score %f', accuracy_scr)

    logger.info('DONE')
