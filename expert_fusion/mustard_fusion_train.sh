jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard.log -err ./mustard.err python blip2_fusion_train.py \
--dataset mustard \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./mustard_blip2_fuser \
--batch_size 10 \
--eval_steps 20 \
--lr 1e-4 \
--epochs 50 \
--max_length 512
