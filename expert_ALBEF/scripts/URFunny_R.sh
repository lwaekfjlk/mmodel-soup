python -m torch.distributed.launch --nproc_per_node=4 --use_env URFunny.py \
--config ./configs/URFunny_R.yaml \
--output_dir ./output/URFunny-R \
--checkpoint ./output/URFunny/checkpoint_best.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env