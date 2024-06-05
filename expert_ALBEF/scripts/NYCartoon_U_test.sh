python -m torch.distributed.launch --nproc_per_node=1 --use_env NYCartoon.py \
--config ./configs/NYCartoon_test.yaml \
--output_dir ./output/NYCartoon-U \
--checkpoint ./output/NYCartoon-U/checkpoint_best.pth \
--no-train