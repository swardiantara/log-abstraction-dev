#!/bin/bash

embeddings=( drone-sbert sbert instructor-base instructor-large instructor-xl )
linkages=( complete average single )
thresholds=( 0.3 0.29 0.28 0.27 0.26 0.25 0.24 0.23 0.22 0.21 )
for embedding in "${embeddings[@]}"; do
    for linkage in "${linkages[@]}"; do
        for threshold in "${thresholds[@]}"; do
            python agglomerative.py --embedding "$embedding" --linkage "$linkage" --threshold "$threshold" --output_dir agglomerative_ajk
        done
    done
done