#! /bin/bash

for split in train dev test; do
# for split in dev; do
    echo Converting $split split...
    python3 scripts/convert_raw_to_gpt.py /mnt/d/Projects/structural-probes/data/ptb-wsj-sd/ptb3-wsj-${split}.txt /mnt/d/Projects/structural-probes/data/ptb-wsj-sd/raw.${split}.gpt2-layers.hdf5 small
done

