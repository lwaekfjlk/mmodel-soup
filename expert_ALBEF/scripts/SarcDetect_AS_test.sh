python -m torch.distributed.launch --nproc_per_node=5 --use_env SarcDetect.py \
--config ./configs/SarcDetect_test.yaml \
--output_dir ./output/SarcDetect-AS \
--checkpoint ./output/SarcDetect-AS/checkpoint_best.pth \
--no-train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env