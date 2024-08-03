jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd.log -err ./mmsd.err python blip2_fusion_train.py \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./mmsd_blip2_fuser \
--batch_size 50 \
--eval_steps 300 \
--epochs 20 \
--load_from_ckpt ../expert_BLIP2/blip2_mmsd_baseline_model \
--max_length 512
