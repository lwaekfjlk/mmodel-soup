#!/bin/bash

python blip2_fusion_train.py \
--mode test \
--test_dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_fuser \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;