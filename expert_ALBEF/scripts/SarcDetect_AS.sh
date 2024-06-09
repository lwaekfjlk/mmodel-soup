python -m torch.distributed.launch --nproc_per_node=5 --use_env SarcDetect.py \
--config ./configs/SarcDetect_AS.yaml \
--output_dir ./output/SarcDetect-AS \
--checkpoint ./ALBEF.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env