#!/bin/sh

python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_AS_model \
--batch_size 30 \
--eval_steps 100 \
--epochs 8 \
--max_length 1024;

python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_R_model \
--batch_size 30 \
--eval_steps 200 \
--epochs 8 \
--max_length 1024;

python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_U_model \
--batch_size 30 \
--eval_steps 200 \
--epochs 8 \
--max_length 1024;

python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_baseline_model \
--batch_size 30 \
--eval_steps 200 \
--epochs 8 \
--max_length 1024;
