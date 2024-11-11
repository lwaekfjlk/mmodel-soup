#!/bin/bash

python blip2_fusion_train_with_classifier.py \
--mode test \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name mmsd_blip2_fuser_with_classifier_A100_multi-task_best_val_loss \
--val_batch_size 60 \
--eval_steps 10 \
--epochs 5 \
--max_length 512