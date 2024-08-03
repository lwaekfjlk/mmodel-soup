python -m torch.distributed.launch --nproc_per_node=4 --use_env MMSD.py \
--config ./configs/MMSD_AS.yaml \
--output_dir ./output/MMSD-AS \
--checkpoint ./output/MMSD/checkpoint_best.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env