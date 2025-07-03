#!/bin/bash
# filepath: /home/x_lab/workspace/roboracer/train_models.sh

# Define arrays for model types, epochs, and learning rates
model_types=(1 2 3)
epochs=(10 20 50)
learning_rates=(0.01 0.001 0.0001)

# Loop through each combination of model type, epoch, and learning rate
for model_type in "${model_types[@]}"; do
  for epoch in "${epochs[@]}"; do
    for lr in "${learning_rates[@]}"; do
      echo "Training with model_type=$model_type, epoch=$epoch, learning_rate=$lr"
      python gpdyn_train.py gp_train.model.type=$model_type gp_train.epoch=$epoch gp_train.learning_rate=$lr
    done
  done
done