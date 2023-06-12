#!/bin/bash
#SBATCH --job-name=e4e
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB


python scripts/train.py \
--dataset_type ffhq_encode \
--exp_dir exps/ffhq_sgxl_1 \
--start_from_latent_avg \
--use_w_pool \
--w_discriminator_lambda 0.1 \
--progressive_start 20000 \
--id_lambda 0.5 \
--val_interval 10000 \
--max_steps 200000 \
--stylegan_size 512 \
--stylegan_weights path/to/pretrained/stylegan.pt \
--workers 8 \
--batch_size 8 \
--test_batch_size 4 \
--test_workers 4 