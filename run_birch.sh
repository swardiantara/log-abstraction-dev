#!/bin/bash

source /home/swardi/miniconda3/bin/activate tencon

embeddings=( instructor-base instructor-large instructor-xl )
datasets=( drone Apache drone_ovs Android BGL Hadoop HDFS HealthApp HPC Linux Mac OpenSSH OpenStack Proxifier Spark Thunderbird Windows Zookeeper )
thresholds=( 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 )
for dataset in "${datasets[@]}"; do
    for embedding in "${embeddings[@]}"; do
        for threshold in "${thresholds[@]}"; do
            python birch.py --dataset "$dataset" --embedding "$embedding" --threshold "$threshold" --output_dir birch-new
        done
    done
done