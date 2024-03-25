#!/bin/bash

read -p "Choose model (1. GPT-2): " model_choice
case $model_choice in 
    1) model="gpt2"
    model_file="GPT"
    ;;
    *) echo "Invalid choice"
    ;;
esac

read -p "Choose layer: " layer

echo "Running experiment for ${model} at layer ${layer}"

python3 structural-probes/run_experiment.py config/${model}/ptb-pad-${model_file}${layer}.yaml

python3 structural-probes/run_experiment.py config/${model}/ptb-prd-${model_file}${layer}.yaml