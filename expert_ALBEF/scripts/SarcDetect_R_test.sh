python -m torch.distributed.launch --nproc_per_node=5 --use_env SarcDetect.py \
--config ./configs/SarcDetect_test.yaml \
--output_dir ./output/SarcDetect-R \
--checkpoint ./output/SarcDetect-R/checkpoint_best.pth \
--no-train