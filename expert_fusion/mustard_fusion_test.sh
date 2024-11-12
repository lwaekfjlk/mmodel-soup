#!/bin/bash

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-test.log -err ./mustard-test.err python blip2_fusion_train.py \
--mode test \
--dataset mustard \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name mustard_blip2_fuser_best_val_loss \
--val_batch_size 20 \
--eval_steps 10 \
--epochs 5 \
--max_length 512
