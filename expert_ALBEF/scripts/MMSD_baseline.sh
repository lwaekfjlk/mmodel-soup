python -m torch.distributed.launch --nproc_per_node=6 --use_env MMSD.py \
--config ./configs/MMSD.yaml \
--output_dir ./output/MMSD \
--checkpoint ./ALBEF.pth \
--train