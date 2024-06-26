#!/bin/bash

embeddings=( drone-sbert sbert instructor-base instructor-large instructor-xl )
thresholds=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )
for embedding in "${embeddings[@]}"; do
    for threshold in "${thresholds[@]}"; do
        python optics.py --embedding "$embedding" --threshold "$threshold" --output_dir optics-ajk
    done
done