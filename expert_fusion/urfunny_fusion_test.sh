#!/bin/bash

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-test.log -err ./urfunny-test.err python blip2_fusion_train.py \
--mode test \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name urfunny_blip2_fuser_50epoch \
--val_batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512