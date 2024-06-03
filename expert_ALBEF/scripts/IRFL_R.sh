python -m torch.distributed.launch --nproc_per_node=1 --use_env IRFL.py \
--config ./configs/IRFL_R.yaml \
--output_dir ./output/IRFL-R \
--checkpoint ./ALBEF.pth \
--train