python blip2_fusion_train.py \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./blip2_fuser \
--batch_size 10 \
--eval_steps 100 \
--epochs 10 \
--max_length 512;
