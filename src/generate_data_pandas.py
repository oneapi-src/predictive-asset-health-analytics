# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

'''
Module to generate dataset for Predictive Asset Health Analytics
'''
# !/usr/bin/env python
# coding: utf-8
import warnings
import argparse
import logging
import time
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-s',
                    '--size',
                    type=int,
                    required=False,
                    default=25000,
                    help='data size')
parser.add_argument('-f',
                    '--file',
                    type=str,
                    required=False,
                    default='asset_data_pandas.pkl',
                    help='output pkl file name')
parser.add_argument('-d',
                    '--debug',
                    action='store_true',
                    help='changes logging level from INFO to DEBUG')

FLAGS = parser.parse_args()
dsize = FLAGS.size

if FLAGS.debug:
    logging_level=logging.DEBUG
else:
    logging_level=logging.INFO

logging.basicConfig(level=logging_level)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Generating our data
start = time.time()
logger.info('Generating data with the size %d', dsize)
np.random.seed(1)
manufacturer_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
species_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
district_list = ['N', 'NE', 'NW', 'E', 'W', 'S', 'SE', 'SW']
treatment_list = ['Oil', 'Pentachlorophenol', 'Untreated', 'Creosote', 'UNK', 'Cellon']
data = pd.DataFrame({"Age": np.random.choice(range(1, 101), dsize, replace=True),
                    "Elevation": np.random.randint(low=-300, high=4500, size=dsize),
                    "Pole_Height": np.random.normal(60, 15, size=dsize),
                    "Measured_Length": np.random.randint(low=1, high=2000, size=dsize),
                    "Manufacturer": np.random.choice(manufacturer_list, dsize, replace=True),
                    "Species": np.random.choice(species_list, dsize, replace=True),
                    "Number_Repairs": np.random.choice(range(1, 7), dsize, replace=True),
                    "District": np.random.choice(district_list, dsize, replace=True),
                    "Tele_Attached": np.random.choice(range(0, 2), dsize, replace=True),
                    "Original_Treatment": np.random.choice(treatment_list, dsize, replace=True)})


# changing Tele_Attatched into an object variable
logger.info('changing Tele_Attatched into an object variable')
data[['Number_Repairs', 'Tele_Attached']] = data[['Number_Repairs', 'Tele_Attached']].astype('object')


# Generating our target variable Asset_Label
logger.info('Generating our target variable Asset_Label')
data['Asset_Label'] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.99, 0.01])


# Creating correlation between our variables and our target variable<br>
# When age is 60-70 and over 95 change Asset_Label to 1
logger.info('Creating correlation between our variables and our target variable')
logger.info('When age is 60-70 and over 95 change Asset_Label to 1')
data['Asset_Label'] = np.where(((data.Age > 0) & (data.Age <= 5)) | (data.Age > 45),
                                1, data.Asset_Label)

# When elevation is between 500-1500 change Asset_Label to 1
logger.info('When elevation is between 500-1500 change Asset_Label to 1')
data['Asset_Label'] = np.where((data.Elevation >= -300) & (data.Elevation <= 1400),
                                1, data.Asset_Label)

# When Manufacturer is A, E, or H change Asset_Label to have  95% 0's
logger.info("When Manufacturer is A, E, or H change Asset_Label to have  95% 0's")
data['Temp_Var'] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.95, 0.05])
data['Asset_Label'] = np.where((data.Manufacturer == 'A') |
                                (data.Manufacturer == 'E') |
                                (data.Manufacturer == 'H'),
                                data.Temp_Var,
                                data.Asset_Label)

# When Species is C2 or C5 change Asset_Label to have 90% to 0's
logger.info("When Species is C2 or C5 change Asset_Label to have 90% to 0's")
data['Temp_Var'] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.9, 0.1])
data['Asset_Label'] = np.where((data.Species == 'C2') | (data.Species == 'C5'),
                                data.Temp_Var,
                                data.Asset_Label)


# When District is NE or W change Asset_Label to have 90% to 0's
logger.info("When District is NE or W change Asset_Label to have 90% to 0's")
data['Temp_Var'] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.95, 0.05])
data['Asset_Label'] = np.where((data.District == 'NE') | (data.District == 'W'),
                                data.Temp_Var, data.Asset_Label)


# When District is Untreated change Asset_Label to have 70% to 1's
logger.info("When District is Untreated change Asset_Label to have 70% to 1's")
data['Temp_Var'] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.25, 0.75])
data['Asset_Label'] = np.where((data.Original_Treatment == 'Untreated'),
                                data.Temp_Var,
                                data.Asset_Label)


# When Age is greater than 90 and Elevation is less than 1200
# and Original_treatment is Oil change Asset_Label to have 90% to 1's
logger.info("When Age is greater than 90 and Elevaation is less than 1200" \
            " and Original_treatment is Oil change Asset_Label to have 90% to 1's")
data['Temp_Var'] = np.random.choice(range(0, 2), dsize, replace=True, p=[0.05, 0.95])
data['Asset_Label'] = np.where((data.Age >= 20) &
                                (data.Elevation <= 1000) &
                                (data.Original_Treatment == 'Oil') &
                                (data.Pole_Height >= 60),
                                data.Temp_Var,
                                data.Asset_Label)

data.drop('Temp_Var', axis=1, inplace=True)


# saving a copy of the dataset for EDA
data.head()

Categorical_Variables = pd.get_dummies(
                            data[[
                                'Manufacturer',
                                'District',
                                'Species',
                                'Original_Treatment']],
                            drop_first=True, dtype='int8')
data = pd.concat([data, Categorical_Variables], axis=1)
data.drop(['Manufacturer', 'District', 'Species', 'Original_Treatment'], axis=1, inplace=True)

data = data.astype({'Tele_Attached': 'int32', 'Number_Repairs': 'float64'})

etime = time.time() - start
datasize = data.shape
logger.info('=====> Time taken %f secs for data generation for the size of %s',
            etime, datasize)


pklfile = FLAGS.file
logger.info('Saving the data to %s ...', pklfile)
data.to_pickle(pklfile)
logger.info('DONE')
