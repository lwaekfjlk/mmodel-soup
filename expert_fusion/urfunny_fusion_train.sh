jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-2e-4lr.log -err ./urfunny-2e-4lr.err python blip2_fusion_train.py \
--dataset urfunny \
--image_data_path ../urfunny_data/data_raw/images \
--save_path ./urfunny_blip2_fuser_2e-4lr \
--batch_size 50 \
--eval_steps 100 \
--epochs 10 \
--lr 2e-4 \
--max_length 512
