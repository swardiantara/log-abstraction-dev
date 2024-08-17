#!/bin/bash

thresholds=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )
embeddings=( drone-sbert sbert simcse instructor-base instructor-large instructor-xl )

for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python birch.py --embedding "$embedding" --threshold "$threshold" --output_dir threshold-analysis
    done
done