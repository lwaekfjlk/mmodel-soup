#!/bin/sh

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-as.log -err ./sarc-as.err  python train.py \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_AS_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--save_path ./blip2_sarc_AS_model \
--load_from_ckpt ./blip2_sarc_baseline_model \
--batch_size 20 \
--eval_steps 100 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-r.log -err ./sarc-r.err python train.py \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_R_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--save_path ./blip2_sarc_R_model \
--load_from_ckpt ./blip2_sarc_baseline_model \
--batch_size 20 \
--eval_steps 200 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-u.log -err ./sarc-u.err python train.py \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_U_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--save_path ./blip2_sarc_U_model \
--load_from_ckpt ./blip2_sarc_baseline_model \
--batch_size 10 \
--eval_steps 200 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc.log -err ./sarc.err python train.py \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--save_path ./blip2_sarc_baseline_model \
--load_from_ckpt ./blip2_sarc_baseline_model \
--batch_size 10 \
--eval_steps 200 \
--epochs 5 \
--max_length 512
