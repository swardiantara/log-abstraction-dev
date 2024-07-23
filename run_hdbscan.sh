#!/bin/bash

source /home/swardi/miniconda3/bin/activate tencon2

embeddings=( drone-sbert sbert simcse )
thresholds=( 0.3 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 )
for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python hdbscan_log.py --embedding "$embedding" --threshold "$threshold" --output_dir hdbscan-ajk
    done
done

source /home/swardi/miniconda3/bin/activate tencon

embeddings=( instructor-base instructor-large instructor-xl )
thresholds=( 0.3 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 )
for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python hdbscan_log.py --embedding "$embedding" --threshold "$threshold" --output_dir hdbscan-ajk
    done
done