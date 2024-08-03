python -m torch.distributed.launch --nproc_per_node=6 --use_env URFunny.py \
--config ./configs/URFunny.yaml \
--output_dir ./output/URFunny \
--checkpoint ./ALBEF.pth \
--train