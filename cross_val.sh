#!/bin/bash

# perform leave-one-out cross-validation
# stdout: Youden's J statistic
# stderr: model name, ground truth, prediction

results_file=$1 # the file containing the model names and distances (ground truth is determined from model name by backdoor() of voting6.py and must be valid for the training set)
ag=$2 # index of the aggregation to use (0: avg, 1: std, 2: min (not used in the paper), 3: max, 4: median)
# optional parameters (used together):
minasr=$3 # minimal ASR required from poisoned models to be included (e.g. 0.25)
weak_file=$4 # the weak_attacks.out file to be used for the ASR filtering

python voting6.py -1 $results_file $ag 0 0 0 1 0 $minasr $weak_file
echo ''
