#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Dataset sizes
SIZE_ARRAY=(25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000)
SIZE_ELEMENTS=${#SIZE_ARRAY[@]}

# Hyperparameter tuning
TUNING_ARRAY=(notuning hyperparametertuning)
TUNING_ELEMENTS=${#TUNING_ARRAY[@]}

# Distribution
DIST_ARRAY=(stock intel)
DIST_ELEMENTS=${#DIST_ARRAY[@]}

PACKAGE="pandas"

for (( i=0;i<$SIZE_ELEMENTS;i++)); do
  echo -e "\t$i. ${SIZE_ARRAY[${i}]}"
done
echo -e "Select dataset size: \c"
read opt
DATASIZE=${SIZE_ARRAY[${opt}]}

for (( i=0;i<$TUNING_ELEMENTS;i++)); do
  echo -e "\t$i. ${TUNING_ARRAY[${i}]}"
done
echo -e "Select tuning option: \c"
read opt
TUNING="0"
if eval "[[ ${TUNING_ARRAY[${opt}]} = ${TUNING_ARRAY[1]} ]]"; then
  TUNING="1"
fi

for (( i=0;i<$DIST_ELEMENTS;i++)); do
  echo -e "\t$i. ${DIST_ARRAY[${i}]}"
done
echo -e "Select xgboost distribution option: \c"
read opt
DIST="0"
if eval "[[ ${DIST_ARRAY[${opt}]} = ${DIST_ARRAY[1]} ]]"; then
  DIST="1"
fi

echo -e "Number of CPU cores to be used for the training: \c"
read opt
CPU=${opt}

echo -e "Creating folder ./$CONDA_DEFAULT_ENV..."
mkdir -p ./$CONDA_DEFAULT_ENV
OF=./$CONDA_DEFAULT_ENV/logfile_${PACKAGE}_${DATASIZE}_$(date +%Y%m%d%H%M%S).log

DATASET=./$CONDA_DEFAULT_ENV/data_$DATASIZE.pkl

## Generate PAM data if it does not exist
if [ ! -e $DATASET ]; then
  echo "+++++++++++++++++++++++++++++++++DataSet++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $OF
  echo -e "Generating Data... \c"
  python3 ./src/generate_data_pandas.py -s $DATASIZE -f $DATASET >> $OF 2>&1
  sync
  echo "DONE"
fi

## First iteration of training and prediction
echo "+++++++++++++++++++++++++++++++++Iteration1++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $OF
echo "[Iteration 1]"
echo -e "Training and Prediction... \c"
python3 ./src/train_predict_pam.py -t $TUNING -p $PACKAGE -f $DATASET -patch $DIST -ncpu $CPU >> $OF 2>&1
sync
echo "DONE"

## Another iteration of training and prediction if it is not meant for hyperparameter tuning
if [ $TUNING = "0" ]; then
    echo "+++++++++++++++++++++++++++++++++Iteration2++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $OF
    echo "[Iteration 2]"
    echo -e "Training and Prediction... \c"
    python3 ./src/train_predict_pam.py -p $PACKAGE -f $DATASET -patch $DIST -ncpu $CPU >> $OF 2>&1
    sync
    echo "DONE"
fi
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $OF

if [ -e data.csv ]; then
  rm data.csv
fi

echo "Logfile stored in $OF"
