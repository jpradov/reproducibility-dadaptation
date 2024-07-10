#!/bin/bash

# Check if n_seeds is provided as a command line argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <n_seeds>"
    exit 1
fi

# Number of seeds to iterate over
n_seeds=$1

# Define constants
data_bin="data-bin/iwslt14.tokenized.de-en"
arch="lstm_wiseman_iwslt_de_en"
dropout=0.3
optimizer="adam"
lr=0.01
adam_betas="(0.9,0.98)"
weight_decay=0.01
max_tokens=12000
max_epoch=55
lr_scheduler="inverse_sqrt"
label_smoothing=0.1
update_freq=1
share_decoder_input_output_embed=""
warmup_updates=4000
log_format="json"

# Iterate over different seeds
for ((i = 1; i <= n_seeds; i++)); do
    seed=$(shuf -i 1-100000 -n 1)

    # Run fairseq-train command with current seed
    fairseq-train "$data_bin" \
          --arch lstm_wiseman_iwslt_de_en \
          --dropout 0.3  \
          --optimizer adam --lr 0.01 --adam-betas '(0.9,0.98)' --weight-decay 0.01\
          --max-tokens 12000 \
          --max-epoch 55 \
          --lr-scheduler inverse_sqrt \
          --fp16 \
          --seed $seed \
          --max-tokens 4096 \
          --criterion label_smoothed_cross_entropy --label-smoothing 0.1\
          --update-freq 1  \
          --share-decoder-input-output-embed \
          --warmup-updates 4000 \
          --log-format "$log_format" > "adam_iwslt14_seed_$seed.json"
done

