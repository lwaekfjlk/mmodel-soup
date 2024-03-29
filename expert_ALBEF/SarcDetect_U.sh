export CUDA_VISIBLE_DEVICES=2

python SarcDetect.py \
--config ./configs/SarcDetect_U.yaml \
--output_dir ./output/SarcDetect-U \
--checkpoint ./output/SarcDetect/checkpoint_best.pth \
--no-train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env