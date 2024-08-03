jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny.log -err ./urfunny.err python blip2_fusion_train.py \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./urfunny_blip2_fuser \
--batch_size 50 \
--eval_steps 70 \
--epochs 50 \
--load_from_ckpt ../expert_BLIP2/blip2_urfunny_baseline_model \
--max_length 512
