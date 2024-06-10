python -m torch.distributed.launch --nproc_per_node=1 --use_env NYCartoon.py \
--config ./configs/NYCartoon_R.yaml \
--output_dir ./output/NYCartoon-R \
--checkpoint ./ALBEF.pth \
--train