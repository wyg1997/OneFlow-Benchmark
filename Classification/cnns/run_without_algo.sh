#!/bin/bash

./train_lenet.sh > log/lenet_WithoutRatio.txt

./train_alexnet.sh > log/alexnet_WithoutRatio.txt

./train_mobilenetV2.sh > log/mobile_WithoutRatio.txt

./train_vgg.sh > log/vgg_WithoutRatio.txt

./train_resnet50.sh > log/resnet50_WithoutRatio.txt

./train_resnext50.sh > log/resnext50_WithoutRatio.txt

