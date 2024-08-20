#!/bin/bash

source /home/swardi/miniconda3/bin/activate tencon2

thresholds=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )
embeddings=( simcse )

for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python birch.py --embedding "$embedding" --threshold "$threshold" --output_dir threshold_birch
    done
done

source /home/swardi/miniconda3/bin/activate tencon

thresholds=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )
embeddings=( instructor-base instructor-large instructor-xl )

for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python birch.py --embedding "$embedding" --threshold "$threshold" --output_dir threshold_birch
    done
done