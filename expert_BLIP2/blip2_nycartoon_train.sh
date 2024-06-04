#!/bin/sh

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_AS_model \
--batch_size 10 \
--eval_steps 300 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_R_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_R_model \
--batch_size 10 \
--eval_steps 300 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_U_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_U_model \
--batch_size 10 \
--eval_steps 300 \
--epochs 5 \
--max_length 512;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_raw/nycartoon_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_baseline_model \
--batch_size 10 \
--eval_steps 300 \
--epochs 5 \
--max_length 512;
