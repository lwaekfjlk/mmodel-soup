python -m torch.distributed.launch --nproc_per_node=4 --use_env URFunny.py \
--config ./configs/URFunny_test.yaml \
--output_dir ./output/URFunny-R \
--checkpoint ./output/URFunny-R/checkpoint_best.pth \
--no-train