#!/bin/bash

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-test.log -err ./mustard-test.err python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-r-test.log -err ./mustard-r-test.err python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-u-test.log -err ./mustard-u-test.err python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-as-test.log -err ./mustard-as-test.err  python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512