#!/bin/bash

embeddings=( simcse )
thresholds=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )
for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python dbscan.py --embedding "$embedding" --threshold "$threshold" --output_dir dbscan-ajk
    done
done