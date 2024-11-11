python -m torch.distributed.launch --nproc_per_node=6 --use_env MMSD.py \
--config ./configs/MMSD_test.yaml \
--output_dir ./output/MMSD-U \
--checkpoint ./output/MMSD-U/checkpoint_best.pth \
--no-train
