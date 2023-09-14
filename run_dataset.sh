#!/bin/bash

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

[[ -z "$WORKSPACE" ]] && { echo "Error: \$WORKSPACE not declared"; exit 1; }
[[ -z "$DATA_DIR" ]] && { echo "Error: \$DATA_DIR not declared"; exit 1; }
[[ -z "$OUTPUT_DIR" ]] && { echo "Error: \$OUTPUT_DIR not declared"; exit 1; }

# Dataset sizes
SIZE_ARRAY=(25000 50000 100000 200000 400000 800000 1000000 2000000 4000000 8000000 10000000)
SIZE_ELEMENTS=${#SIZE_ARRAY[@]}

# Hyperparameter tuning
TUNING_ARRAY=(notuning hyperparametertuning)
TUNING_ELEMENTS=${#TUNING_ARRAY[@]}

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

echo -e "Number of CPU cores to be used for the training: \c"
read opt
CPU=${opt}

mkdir -p $OUTPUT_DIR/logs

mkdir -p $DATA_DIR
DATASET=$DATA_DIR/data_$DATASIZE.pkl

## Generate PAM data if it does not exist
if [ ! -e $DATASET ]; then
  OF=$OUTPUT_DIR/logs/logfile_pandas_${DATASIZE}_$(date +%Y%m%d%H%M%S).log
  echo "+++++++++++++++++++++++++++++++++DataSet++++++++++++++++++++++++++++++++++++++++++++++++++++" >> $OF
  echo -e "Generating Data... \n"
  python3 $WORKSPACE/src/generate_data_pandas.py -s $DATASIZE -f $DATASET 2>&1 | tee $OF 
  echo "Dataset Generation logfile stored in $OF"
  sync
  echo "Generating Data: DONE"
fi

## Training and prediction
OF=$OUTPUT_DIR/logs/logfile_train_predict_${DATASIZE}_$(date +%Y%m%d%H%M%S).log
echo "++++++++++++++++++++++++++Training and Prediction++++++++++++++++++++++++++++++++++++++++++++" >> $OF
echo -e "Training and Prediction... \n"
python3 $WORKSPACE/src/train_predict_pam.py -t $TUNING -p "pandas" -f $DATASET -ncpu $CPU 2>&1 | tee -a $OF 
sync
echo "Dataset stored in $DATASET"
echo "Train Prediction logfile stored in $OF"
echo "Training and Prediction: DONE"
