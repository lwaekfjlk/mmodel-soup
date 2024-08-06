#!/bin/bash

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-test-${SEED}.log -err ./urfunny-test-${SEED}.err python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-r-test-${SEED}.log -err ./urfunny-r-test-${SEED}.err python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_R_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-u-test-${SEED}.log -err ./urfunny-u-test-${SEED}.err python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_U_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-as-test-${SEED}.log -err ./urfunny-as-test-${SEED}.err  python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_AS_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512