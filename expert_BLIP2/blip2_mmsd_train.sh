#!/bin/sh

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 \
-out ./mmsd-as-seed-${SEED}.log \
-err ./mmsd-as-seed-${SEED}.err \
python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_AS_model_${SEED} \
--load_from_ckpt ./blip2_mmsd_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512 \
--seed ${SEED}

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 \
-out ./mmsd-r-seed-${SEED}.log \
-err ./mmsd-r-seed-${SEED}.err \
python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_R_model_${SEED} \
--load_from_ckpt ./blip2_mmsd_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512 \
--seed ${SEED}

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 \
-out ./mmsd-u-seed-${SEED}.log \
-err ./mmsd-u-seed-${SEED}.err \
python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_split_output/mmsd_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../mmsd_data/data_split_output/mmsd_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_U_model_${SEED} \
--load_from_ckpt ./blip2_mmsd_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512 \
--seed ${SEED}

export SEED=3456;
jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 \
-out ./mmsd-seed-${SEED}.log \
-err ./mmsd-seed-${SEED}.err \
python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./blip2_mmsd_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 100 \
--epochs 4 \
--max_length 512 \
--seed ${SEED}
