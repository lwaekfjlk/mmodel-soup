#!/bin/bash

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-test.log -err ./mmsd-test.err python blip2_fusion_train.py \
--mode test \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name mmsd_blip2_fuser_best_combined_precision \
--val_batch_size 20 \
--eval_steps 10 \
--epochs 5 \
--max_length 512