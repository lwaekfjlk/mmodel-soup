python -m torch.distributed.launch --nproc_per_node=1 --use_env IRFL.py \
--config ./configs/IRFL_AS.yaml \
--output_dir ./output/IRFL-AS \
--checkpoint ./ALBEF.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env