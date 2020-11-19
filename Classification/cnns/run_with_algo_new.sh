#!/bin/bash

for RatioValue in 0.01 0.1 1 10 100 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_lenet.sh > log/lenet_WithRatio${RatioValue}.txt
done 

for RatioValue in 0.01 0.1 1 10 100 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_alexnet.sh > log/alexnet_WithRatio${RatioValue}.txt
done 

for RatioValue in 0.01 0.1 1 10 100 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_mobilenetV2.sh > log/mobile_WithRatio${RatioValue}.txt
done 

for RatioValue in 0.01 0.1 1 10 100 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_vgg.sh > log/vgg_WithRatio${RatioValue}.txt
done 

for RatioValue in 0.01 0.1 1 10 100 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_resnet50.sh > log/resnet50_WithRatio${RatioValue}.txt
done 

for RatioValue in 0.01 0.1 1 10 100 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_resnext50.sh > log/resnext50_WithRatio${RatioValue}.txt
done 
