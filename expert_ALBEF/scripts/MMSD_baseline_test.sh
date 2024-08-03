python -m torch.distributed.launch --nproc_per_node=4 --use_env MMSD.py \
--config ./configs/MMSD_test.yaml \
--output_dir ./output/MMSD \
--checkpoint ./output/MMSD/checkpoint_best.pth \
--no-train