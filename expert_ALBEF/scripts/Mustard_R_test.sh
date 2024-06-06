python -m torch.distributed.launch --nproc_per_node=1 --use_env Mustard.py \
--config ./configs/Mustard_test.yaml \
--output_dir ./output/Mustard-R \
--checkpoint ./output/Mustard-R/checkpoint_best.pth \
--no-train