python -m torch.distributed.launch --nproc_per_node=4 --use_env MMSD.py \
--config ./configs/MMSD_test.yaml \
--output_dir ./output/MMSD-R \
--checkpoint ./output/MMSD-R/checkpoint_best.pth \
--no-train