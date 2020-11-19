#!/bin/bash

for RatioValue in 1e-7 1e-8 1e-9 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_lenet.sh > log/lenet_WithRatio${RatioValue}.txt
done 
