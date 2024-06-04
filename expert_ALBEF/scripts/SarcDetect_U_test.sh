python -m torch.distributed.launch --nproc_per_node=4 --use_env SarcDetect.py \
--config ./configs/SarcDetect_test.yaml \
--output_dir ./output/SarcDetect-U-test \
--checkpoint ./output/SarcDetect-U/checkpoint_best.pth \
--no-train