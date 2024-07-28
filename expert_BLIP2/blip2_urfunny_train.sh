#!/bin/sh

python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_AS_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 8 \
--max_length 512;

python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_R_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 8 \
--max_length 512;

python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_U_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 8 \
--max_length 512;

python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_raw/urfunny_dataset_train.json \
--val_path ../urfunny_data/data_raw/urfunny_dataset_val.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 8 \
--max_length 512;
