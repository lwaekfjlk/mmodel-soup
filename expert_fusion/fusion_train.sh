python blip2_fusion_train.py \
--dataset mustard \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./blip2_mustard_AS_model \
--batch_size 2 \
--eval_steps 60 \
--epochs 10 \
--max_length 512;
