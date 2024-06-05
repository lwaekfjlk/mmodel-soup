python -m torch.distributed.launch --nproc_per_node=1 --use_env NYCartoon.py \
--config ./configs/NYCartoon_AS.yaml \
--output_dir ./output/NYCartoon-AS \
--checkpoint ./ALBEF.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env