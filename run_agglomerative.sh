#!/bin/bash

embeddings=( drone-sbert sbert instructor-base instructor-large instructor-xl )
linkages=( complete average single )
thresholds=( 0.2 0.19 0.18 0.17 0.16 0.15 0.14 0.13 0.12 0.11 0.1 0.09 0.08 0.07 0.06 0.05 0.04 0.03 0.02 0.01 )
for embedding in "${embeddings[@]}"; do
    for linkage in "${linkages[@]}"; do
        for threshold in "${thresholds[@]}"; do
            python agglomerative.py --embedding "$embedding" --linkage "$linkage" --threshold "$threshold" --output_dir agglomerative_ajk
        done
    done
done