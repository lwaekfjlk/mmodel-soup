#!/bin/sh

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-as.log -err ./urfunny-as.err  python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_AS_model \
--load_from_ckpt ./blip2_urfunny_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-r.log -err ./urfunny-r.err python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_R_model \
--load_from_ckpt ./blip2_urfunny_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-u.log -err ./urfunny-u.err python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_split_output/urfunny_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../urfunny_data/data_split_output/urfunny_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_U_model \
--load_from_ckpt ./blip2_urfunny_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny.log -err ./urfunny.err python train.py \
--dataset urfunny \
--train_path ../urfunny_data/data_raw/urfunny_dataset_train.json \
--val_path ../urfunny_data/data_raw/urfunny_dataset_val.json \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_urfunny_baseline_model \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--max_length 512
