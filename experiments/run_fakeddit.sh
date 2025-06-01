#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export CUDA_VISIBLE_DEVICES=0

# Create necessary directories
mkdir -p logs checkpoints

# Run training
python training/train.py \
    --config config/default.yaml \
    --data_dir data/processed/fakeddit \
    --output_dir checkpoints \
    --log_dir logs \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4

# Run evaluation
python evaluation/evaluate.py \
    --model_path checkpoints/best_model.pt \
    --data_dir data/processed/fakeddit \
    --output_dir evaluation 