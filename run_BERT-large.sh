# read the text from text_data.txt
data=$(cat text_data.txt)

# read one line at a time
while read line; do
    sentences=$(awk -F'\\| ' '{s=""; for (i=1; i<=NF; i++) {s=s $i; print s}}' <<< "$line")
    printf '%s\n' "$sentences" | python3 structural-probes/run_demo.py example/demo-bert.yaml
done <<< "$data"