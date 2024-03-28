#!/bin/bash

# This script generates the config files based on the file for layer 0

cd config/roberta-large 

model="ROBERTAlarge"
num_layer=24

for i in $(seq 1 $num_layer); do
    cat ptb-pad-${model}0.yaml | sed "s/layer: 0/layer: $i/g" > ptb-pad-${model}$i.yaml
    cat ptb-prd-${model}0.yaml | sed "s/layer: 0/layer: $i/g" > ptb-prd-${model}$i.yaml
done