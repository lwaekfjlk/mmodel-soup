#!/bin/sh

export SEED=3333;
python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_raw/urfunny_dataset_train.json \
--val_path ../urfunny_data/data_raw/urfunny_dataset_val.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--seed ${SEED} \
--max_length 512

export SEED=3333;
python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_AS_model_${SEED} \
--load_from_ckpt ./blip2_urfunny_baseline_model_1234 \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--seed ${SEED} \
--max_length 512

export SEED=3333;
python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_R_model_${SEED} \
--load_from_ckpt ./blip2_urfunny_baseline_model_1234 \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--seed ${SEED} \
--max_length 512

export SEED=3333;
python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_U_model_${SEED} \
--load_from_ckpt ./blip2_urfunny_baseline_model_1234 \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--seed ${SEED} \
--max_length 512
