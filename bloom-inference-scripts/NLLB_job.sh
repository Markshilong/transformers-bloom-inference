#!/bin/bash
#SBATCH -o NLLB.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -J MoEjob
nvidia-smi
cd /home/shilong/transformers-bloom-inference
deepspeed --num_gpus 1 bloom-inference-scripts/bloom-ds-zero-inference.py --name facebook/nllb-moe-54b --cpu_offload