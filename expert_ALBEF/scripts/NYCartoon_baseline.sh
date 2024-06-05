python -m torch.distributed.launch --nproc_per_node=1 --use_env NYCartoon.py \
--config ./configs/NYCartoon.yaml \
--output_dir ./output/NYCartoon \
--checkpoint ./ALBEF.pth \
--train