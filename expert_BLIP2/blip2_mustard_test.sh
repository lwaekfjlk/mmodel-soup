#!/bin/bash

export SEED=1;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-test-${SEED}.log -err ./mustard-test-${SEED}.err python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-r-test-${SEED}.log -err ./mustard-r-test-${SEED}.err python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_R_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-u-test-${SEED}.log -err ./mustard-u-test-${SEED}.err python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_U_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-as-test-${SEED}.log -err ./mustard-as-test-${SEED}.err  python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_AS_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512
