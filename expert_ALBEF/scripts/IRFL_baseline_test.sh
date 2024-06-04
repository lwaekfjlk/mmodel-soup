python -m torch.distributed.launch --nproc_per_node=1 --use_env IRFL.py \
--config ./configs/IRFL_test.yaml \
--output_dir ./output/IRFL \
--checkpoint ./output/IRFL/checkpoint_best.pth \
--no-train