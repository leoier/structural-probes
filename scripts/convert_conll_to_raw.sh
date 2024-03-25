#!/bin/bash

for split in train dev test; do
    echo Converting $split split...
    python3 scripts/convert_conll_to_raw.py ptb3-wsj-${split}.conllx > ptb3-wsj-${split}.txt
done

