python -m torch.distributed.launch --nproc_per_node=1 --use_env NYCartoon.py \
--config ./configs/NYCartoon_U.yaml \
--output_dir ./output/NYCartoon-U \
--checkpoint ./ALBEF.pth \
--train