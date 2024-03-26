#!/bin/bash
# Generates the parse trees using the GPT-2 model

# read the text from text_data.txt
data=$(cat text_data.txt)


# contruct the parse trees for each sentence incrementally
# with comma
cnt=1
while read line; do
    sentences=$(awk -F'\\| ' '{s=""; for (i=1; i<=NF; i++) {s=s $i; print s}}' <<< "$line")
    printf '%s\n' "$sentences" | python3 structural-probes/run_demo_gpt2.py demo-gpt2.yaml
    # find the last folder start with gpt-2-demo-
    folder=$(find results/gpt2/ -type d -name "gpt-2-demo-*" | sort | tail -n 1)
    cp ${folder}/demo.tikz results/gpt2/gpt-2-parse-trees/comma/${cnt}.tikz
    rm -r ${folder}
    cnt=$((cnt+1))
done <<< "$data"

# without comma
data=$(echo "${data//,}")
cnt=1
while read line; do
    sentences=$(awk -F'\\| ' '{s=""; for (i=1; i<=NF; i++) {s=s $i; print s}}' <<< "$line")
    printf '%s\n' "$sentences" | python3 structural-probes/run_demo_gpt2.py demo-gpt2.yaml
    # find the last folder start with gpt-2-demo-
    folder=$(find results/gpt2/ -type d -name "gpt-2-demo-*" | sort | tail -n 1)
    cp ${folder}/demo.tikz results/gpt2/gpt-2-parse-trees/no_comma/${cnt}.tikz
    rm -r ${folder}
    cnt=$((cnt+1))
done <<< "$data"
