jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-focal-loss.log -err ./urfunny-focal-loss.err python blip2_fusion_train.py \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./urfunny_blip2_fuser_focal_loss \
--batch_size 50 \
--eval_steps 70 \
--epochs 50 \
--max_length 512
