#!/bin/sh

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_metaphor_AS_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_metaphor_AS_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_metaphor_R_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_metaphor_R_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_metaphor_U_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_metaphor_U_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_metaphor_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_metaphor_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./blip2_irfl_metaphor_baseline_model \
--batch_size 10 \
--eval_steps 150 \
--epochs 5 \
--max_length 512;