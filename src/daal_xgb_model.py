# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Module to convert XGBoost trained model to optimized daal4py version
'''
# !/usr/bin/env python
# coding: utf-8
import time
import warnings
# import matplotlib.pyplot as plt
import logging
from typing import Deque, Dict, Any
from collections import deque
import json
import argparse
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import daal4py as d4p
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()
parser.add_argument('-f',
                    '--file',
                    type=str,
                    required=False,
                    default='data_25000.pkl',
                    help='input pkl file name')
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

start = time.time()
logger.info('Reading the dataset from %s...', FLAGS.file)
try:
    data = pd.read_pickle(FLAGS.file)
except FileNotFoundError:
    sys.exit('Dataset file not found')

data['Original_Treatment_Untreated'].describe()

datasize = data.shape

X = data.drop('Asset_Label', axis=1)
y = data.Asset_Label

X = X.rename(columns={x: y for x, y in zip(X.columns, range(0, len(X.columns)))})  # pylint: disable=unnecessary-comprehension

X[4] = X[4].astype('int32')

# original split .25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

# Datasets creation
xgb_train = xgb.DMatrix(X_train, label=np.array(y_train))
xgb_test = xgb.DMatrix(X_test, label=np.array(y_test))

train_start = time.time()
# training parameters setting
params = {
    'max_bin': 256,
    'scale_pos_weight': 2,
    'lambda_l2': 1,
    'alpha': 0.9,
    'max_depth': 8,
    'num_leaves': 2**8,
    'verbosity': 0,
    'objective': 'multi:softmax',
    'learning_rate': 0.3,
    'num_class': 5,
}

# Training
xgb_model = xgb.train(params, xgb_train, num_boost_round=100)

total_train_time = time.time() - train_start
logger.info('XGBoost training time (seconds): %f', total_train_time)

props = dict(boxstyle='round', facecolor='cyan', alpha=0.5)

# Training - Training Time Benchmark
left = [1]

rounded_train_time = round(total_train_time, 5)

tick_label = ['XGBoost Training Model']

# XGBoost prediction (for accuracy comparison)
xgb_start_time = time.time()
xgb_prediction = xgb_model.predict(xgb_test)
xgb_total = time.time() - xgb_start_time

xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(y_test))

logger.info('XGBoost inference time (seconds): %f', xgb_total)


# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
def get_gbt_model_from_xgboost(booster: Any) -> Any:  # pylint: disable=too-many-statements
    '''
    Class Node
    '''
    class Node:  # pylint: disable=too-few-public-methods
        """Class representing a Node"""
        def __init__(self, tree: Dict, parent_id: int, position: int):
            self.tree = tree
            self.parent_id = parent_id
            self.position = position

    # Release Note for XGBoost 1.5.0: Python interface now supports configuring
    # constraints using feature names instead of feature indices.
    if booster.feature_names is None:
        lst = [*range(booster.num_features())]
        booster.feature_names = [str(i) for i in lst]
    # constraints using feature names instead of feature indices. This also
    # helps with pandas input with set feature names.
    lst = [*range(booster.num_features())]
    booster.feature_names = [str(i) for i in lst]

    trees_arr = booster.get_dump(dump_format="json")
    xgb_config = json.loads(booster.save_config())
    n_features = int(xgb_config["learner"]["learner_model_param"]["num_feature"])
    n_classes = int(xgb_config["learner"]["learner_model_param"]["num_class"])
    base_score = float(xgb_config["learner"]["learner_model_param"]["base_score"])
    is_regression = False
    objective_fun = xgb_config["learner"]["learner_train_param"]["objective"]
    if n_classes > 2:
        if objective_fun not in ["multi:softprob", "multi:softmax"]:
            raise TypeError(
                "multi:softprob and multi:softmax are only supported for multiclass classification")
    elif objective_fun.find("binary:") == 0:
        if objective_fun in ["binary:logistic", "binary:logitraw"]:
            n_classes = 2
        else:
            raise TypeError(
                "binary:logistic and binary:logitraw are only supported for binary classification")
    else:
        is_regression = True
    n_iterations = int(len(trees_arr) / (n_classes if n_classes > 2 else 1))
    # Create + base iteration
    if is_regression:
        m_b = d4p.gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations + 1)  # pylint: disable=undefined-variable
        tree_id = m_b.create_tree(1)
        m_b.add_leaf(tree_id=tree_id, response=base_score)
    else:
        m_b = d4p.gbt_clf_model_builder(  # pylint: disable=no-member
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes)
    class_label = 0
    iterations_counter = 0
    mis_eq_yes = None
    for tree in trees_arr:
        n_nodes = 1
        # find out the number of nodes in the tree
        for node in tree.split("nodeid")[1:]:
            node_id = int(node[3:node.find(",")])
            if node_id + 1 > n_nodes:
                n_nodes = node_id + 1
        if is_regression:
            tree_id = m_b.create_tree(n_nodes)
        else:
            tree_id = m_b.create_tree(n_nodes=n_nodes, class_label=class_label)
        iterations_counter += 1
        if iterations_counter == n_iterations:
            iterations_counter = 0
            class_label += 1
        sub_tree = json.loads(tree)
        # root is leaf
        if "leaf" in sub_tree:
            m_b.add_leaf(tree_id=tree_id, response=sub_tree["leaf"])
            continue
        # add root
        try:
            feature_index = int(sub_tree["split"])
        except ValueError as typeerror:
            raise TypeError("Feature names must be integers") from typeerror
        feature_value = np.nextafter(np.single(sub_tree["split_condition"]), np.single(-np.inf))
        parent_id = m_b.add_split(tree_id=tree_id, feature_index=feature_index,
                                  feature_value=feature_value)
        # create queue
        yes_idx = sub_tree["yes"]
        no_idx = sub_tree["no"]
        mis_idx = sub_tree["missing"]
        if mis_eq_yes is None:
            if mis_idx == yes_idx:
                mis_eq_yes = True
            elif mis_idx == no_idx:
                mis_eq_yes = False
            else:
                raise TypeError(
                    "Missing values are not supported in daal4py Gradient Boosting Trees")
        elif mis_eq_yes and mis_idx != yes_idx or not mis_eq_yes and mis_idx != no_idx:
            raise TypeError("Missing values are not supported in daal4py Gradient Boosting Trees")
        node_queue: Deque[Node] = deque()  # pylint: disable=undefined-variable
        node_queue.append(Node(sub_tree["children"][0], parent_id, 0))
        node_queue.append(Node(sub_tree["children"][1], parent_id, 1))
        # bfs through it
        while node_queue:
            sub_tree = node_queue[0].tree
            parent_id = node_queue[0].parent_id
            position = node_queue[0].position
            node_queue.popleft()
            # current node is leaf
            if "leaf" in sub_tree:
                m_b.add_leaf(
                    tree_id=tree_id, response=sub_tree["leaf"],
                    parent_id=parent_id, position=position)
                continue
            # current node is split
            try:
                feature_index = int(sub_tree["split"])
            except ValueError as typeerror:
                raise TypeError("Feature names must be integers") from typeerror
            feature_value = np.nextafter(np.single(sub_tree["split_condition"]), np.single(-np.inf))
            parent_id = m_b.add_split(
                tree_id=tree_id, feature_index=feature_index, feature_value=feature_value,
                parent_id=parent_id, position=position)
            # append to queue
            yes_idx = sub_tree["yes"]
            no_idx = sub_tree["no"]
            mis_idx = sub_tree["missing"]
            if mis_eq_yes and mis_idx != yes_idx or not mis_eq_yes and mis_idx != no_idx:
                raise TypeError(
                    "Missing values are not supported in daal4py Gradient Boosting Trees")
            node_queue.append(Node(sub_tree["children"][0], parent_id, 0))
            node_queue.append(Node(sub_tree["children"][1], parent_id, 1))
    return m_b.model()


# Conversion to daal4py
daal_conv_stime = time.time()
daal_model = d4p.get_gbt_model_from_xgboost(xgb_model)  # pylint: disable=no-member
daal_conv_etime = time.time()

daal_conv_total = daal_conv_etime - daal_conv_stime
logger.info('DAAL conversion time (seconds): %f', daal_conv_total)

# daal4py prediction
daal_predict_algo = d4p.gbt_classification_prediction(  # pylint: disable=no-member
    nClasses=params["num_class"],
    resultsToEvaluate="computeClassLabels",
    fptype='float'
)
daal_start_time = time.time()
daal_prediction = daal_predict_algo.compute(X_test, daal_model)
d4p_total = time.time() - daal_start_time

daal_errors_count = np.count_nonzero(daal_prediction.prediction[:, 0] - np.ravel(y_test))
#logger.info(daal_errors_count)

logger.info('DAAL inference time (seconds): %f', d4p_total)

logger.info("XGBoost errors count: %d", xgb_errors_count)
xgb_acc = abs((xgb_errors_count / xgb_prediction.shape[0]) - 1)
logger.info("XGBoost accuracy: %f", xgb_acc)

logger.info("Daal4py errors count: %d", daal_errors_count)
d4p_acc = abs((daal_errors_count / xgb_prediction.shape[0]) - 1)
logger.info("Daal4py accuracy: %f", d4p_acc)

logger.info("XGBoost Prediction Time: %f", xgb_total)
logger.info("daal4py Prediction Time: %f", d4p_total)

# Performance - Prediction Time
rounded_xgb = round(xgb_total, 4)
rounded_daal = round(d4p_total, 4)

left = [1, 2]
pred_times = [rounded_xgb, rounded_daal]
tick_label = ['XGBoost Prediction', 'daal4py Prediction']

# Performance - Prediction Time Benchmark
left = [1]
perf_bench = abs((d4p_total/xgb_total) - 1)

tick_label = ['daal4py Prediction']

logger.info("daal4py time improvement relative to XGBoost: %f", perf_bench)

# Accurancy
left = [1, 2]
xgb_acc = abs((xgb_errors_count / xgb_prediction.shape[0]) - 1)

d4p_acc = abs((daal_errors_count / xgb_prediction.shape[0]) - 1)

pred_acc = [xgb_acc, d4p_acc]
tick_label = ['XGBoost Prediction', 'daal4py Prediction']

logger.info("Accuracy Difference %f", xgb_acc-d4p_acc)
