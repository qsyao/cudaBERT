# Retraining

Add Layer ： Linear(hiddensize, num_classes) + Softmax(dim=-1)

## Before Run Test:

Generate Weights and bias(npy) to model_npy like branch master

## Compile

cmake .
make

## Run test
./unit_test