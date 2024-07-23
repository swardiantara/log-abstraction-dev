#!/bin/bash

thresholds=( 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 )

for threshold in "${thresholds[@]}"; do
    python agglomerative.py --embedding drone-sbert --linkage average --threshold "$threshold" --output_dir threshold_agglomerative
done

for threshold in "${thresholds[@]}"; do
    python birch.py --embedding drone-sbert --threshold "$threshold" --output_dir threshold_birch
done

for threshold in "${thresholds[@]}"; do
    python dbscan.py --embedding drone-sbert --threshold "$threshold" --output_dir threshold_dbscan
done

for threshold in "${thresholds[@]}"; do
    python hdbscan_log.py --embedding drone-sbert --threshold "$threshold" --output_dir threshold_hdbscan
done

for threshold in "${thresholds[@]}"; do
    python optics.py --embedding drone-sbert --threshold "$threshold" --output_dir threshold_optics
done