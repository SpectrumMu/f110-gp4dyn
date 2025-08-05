#!/bin/bash
# filepath: /home/x_lab/workspace/roboracer/train_models.sh

# Define arrays for model types, epochs, and learning rates
model_types=(1)
# epochs=(10 20 50)
# learning_rates=(0.01 0.001 0.0001)
inducing=(500 1000 1500 2000)
if_normalization=(true)

# Loop through each combination of model type, epoch, and learning rate
for model_type in "${model_types[@]}"; do
  for inducing_size in "${inducing[@]}"; do
    for normalization in "${if_normalization[@]}"; do
      echo "Training with model_type=$model_type, inducing_size=$inducing_size, if_normalization=$normalization"
      # Run the training script with the specified parameters
      python gpdyn_train.py gp_train.model.type=$model_type gp_train.model.inducing=$inducing_size gp_train.if_norm=$normalization
    done
  done
  # for epoch in "${epochs[@]}"; do
    # for lr in "${learning_rates[@]}"; do
      # echo "Training with model_type=$model_type, epoch=$epoch, learning_rate=$lr"
      # python gpdyn_train.py gp_train.model.type=$model_type gp_train.epoch=$epoch gp_train.learning_rate=$lr
    # done
  # done
done