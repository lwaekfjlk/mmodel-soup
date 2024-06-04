python SarcDetect.py \
--config ./configs/SarcDetect.yaml \
--output_dir ./output/SarcDetect-test \
--checkpoint ./output/SarcDetect/checkpoint_best.pth \
--no-train 
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env 