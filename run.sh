#!bin/bash

CUDA_VISIBLE_DEVICES=1 python Simulation.py --config DatasetConfig.yaml
CUDA_VISIBLE_DEVICES=1 python Simulation.py --config DatasetConfig_2.yaml
CUDA_VISIBLE_DEVICES=1 python Simulation.py --config DatasetConfig_5.yaml
CUDA_VISIBLE_DEVICES=1 python Simulation.py --config DatasetConfig_6.yaml