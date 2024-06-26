#!/bin/sh

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_AS_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_AS_model \
--batch_size 10 \
--eval_steps 300 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_R_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_R_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_U_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_U_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_baseline_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;