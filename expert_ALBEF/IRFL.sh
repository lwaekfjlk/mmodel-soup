python IRFL.py \
--config ./configs/IRFL.yaml \
--output_dir ./output/IRFL-test \
--checkpoint ./output/IRFL/checkpoint_best.pth \
--train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env 