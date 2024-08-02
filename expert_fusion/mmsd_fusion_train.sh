jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-2e-4lr.log -err ./mmsd-2e-4lr.err python blip2_fusion_train.py \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./mmsd_blip2_fuser_2e-4lr \
--batch_size 50 \
--eval_steps 300 \
--lr 2e-4 \
--epochs 50 \
--max_length 512