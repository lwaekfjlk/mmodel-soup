python -m torch.distributed.launch --nproc_per_node=1 --use_env IRFL.py \
--config ./configs/IRFL.yaml \
--output_dir ./output/IRFL \
--checkpoint ./ALBEF.pth \
--train