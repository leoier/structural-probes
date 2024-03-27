#! /bin/bash

read -p "Choose the model (1. GPT-2, 2. RoBERTa-base, 3. RoBERTa-large): " model_idx
case $model_idx in
    1)
        model="gpt2"
        ;;
    2)
        model="roberta-base"
        ;;
    3)
        model="roberta-large"
        ;;
    *)
        echo "Invalid model index."
        exit 1
        ;;
esac


for split in train dev test; do
    echo Converting $split split...
    python3 scripts/convert_raw_to_hidden.py /mnt/d/Projects/structural-probes/data/ptb-wsj-sd/ptb3-wsj-${split}.txt /mnt/d/Projects/structural-probes/data/ptb-wsj-sd/raw.${split}.${model}-layers.hdf5 ${model}
done

