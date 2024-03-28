#!/bin/bash

read -p "Choose model (1. GPT-2, 2. RoBERTa-base, 3. RoBERTa-large): " model_choice
case $model_choice in 
    1) model="gpt2"
    model_file="GPT"
    ;;
    2) model="roberta-base"
    model_file="ROBERTAbase"
    num_layer=12
    ;;
    3) model="roberta-large"
    model_file="ROBERTAlarge"
    num_layer=24
    ;;
    *) echo "Invalid choice"
    ;;
esac

read -p "Choose layer (-1 for all layers): " layer

if [ ${layer} -ge 0 ]; then
    layers=[${layer}]
else
    layers=$(seq 0 ${num_layer})
fi


for layer in ${layers}; do
    echo "Training depth probe for ${model} at layer ${layer}"
    python3 structural-probes/run_experiment.py config/${model}/ptb-pad-${model_file}${layer}.yaml
    echo "Training distance probe for ${model} at layer ${layer}"
    python3 structural-probes/run_experiment.py config/${model}/ptb-prd-${model_file}${layer}.yaml
done