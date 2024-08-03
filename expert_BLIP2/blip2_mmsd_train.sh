#!/bin/sh

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-as.log -err ./mmsd-as.err  python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_AS_model \
--load_from_ckpt ./blip2_mmsd_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-r.log -err ./mmsd-r.err python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_R_model \
--load_from_ckpt ./blip2_mmsd_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-u.log -err ./mmsd-u.err python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_U_model \
--load_from_ckpt ./blip2_mmsd_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd.log -err ./mmsd.err python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--max_length 512
