#!/bin/bash
#SBATCH --job-name=e4e
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB

batch_size=8
workers=4

name=$1

export CUDA_HOME="/pkgs/cuda-11.2"

python scripts/train.py \
--dataset_type ffhq_encode \
--exp_dir exps/$name \
--start_from_latent_avg \
--use_w_pool \
--w_discriminator_lambda 0.1 \
--progressive_start 20000 \
--id_lambda 0.5 \
--val_interval 10000 \
--save_interval 500 \
--max_steps 200000 \
--stylegan_size 256 \
--stylegan_weights ~/stylegan-xl/pretrained_models/ffhq256.pkl \
--workers $workers \
--batch_size $batch_size \
--test_batch_size $batch_size \
--test_workers $workers \
--ckpt_path /checkpoint/kaselby/$name/ckpt.pt \
--save_training_data \
--stem_size 16 \
--syn_layers 10 \
--head_layers 4 \
--sgxl
