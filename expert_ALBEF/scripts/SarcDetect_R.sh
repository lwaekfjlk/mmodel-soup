python -m torch.distributed.launch --nproc_per_node=5 --use_env SarcDetect.py \
--config ./configs/SarcDetect_R.yaml \
--output_dir ./output/SarcDetect-R \
--checkpoint ./ALBEF.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env