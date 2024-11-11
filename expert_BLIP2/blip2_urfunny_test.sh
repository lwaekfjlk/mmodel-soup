#!/bin/bash

export SEED=1111;
python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=1111;
python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_R_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=1111;
python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_U_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=1111;
python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_AS_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512