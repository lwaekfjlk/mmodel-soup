python -m torch.distributed.launch --nproc_per_node=1 --use_env IRFL.py \
--config ./configs/IRFL_U.yaml \
--output_dir ./output/IRFL-U \
--checkpoint ./ALBEF.pth \
--train