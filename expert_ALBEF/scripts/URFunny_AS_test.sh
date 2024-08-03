python -m torch.distributed.launch --nproc_per_node=6 --use_env URFunny.py \
--config ./configs/URFunny_test.yaml \
--output_dir ./output/URFunny-AS \
--checkpoint ./output/URFunny-AS/checkpoint_best.pth \
--no-train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env