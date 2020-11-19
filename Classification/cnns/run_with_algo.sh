#!/bin/bash

for RatioValue in 1e-7 1e-8 1e-9 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_lenet.sh > log/lenet_WithRatio${RatioValue}.txt
done 

for RatioValue in 1e-7 1e-8 1e-9 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_alexnet.sh > log/alexnet_WithRatio${RatioValue}.txt
done 

for RatioValue in 1e-7 1e-8 1e-9 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_mobilenetV2.sh > log/mobile_WithRatio${RatioValue}.txt
done 

for RatioValue in 1e-7 1e-8 1e-9 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_vgg.sh > log/vgg_WithRatio${RatioValue}.txt
done 

for RatioValue in 1e-6 1e-7 1e-8 1e-9 1e-10; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_resnet50.sh > log/resnet50_WithRatio${RatioValue}.txt
done 

for RatioValue in 1e-7 1e-8 1e-9 ; do
    echo "${RatioValue}" > CostRatioFile.txt
    ./train_resnext50.sh > log/resnext50_WithRatio${RatioValue}.txt
done 
