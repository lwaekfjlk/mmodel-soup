#!/bin/bash

export SEED=3456;
python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3456;
python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_R_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-u-test-${SEED}.log -err ./mmsd-u-test-${SEED}.err python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_U_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3456;
python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_AS_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512