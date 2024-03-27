python -m torch.distributed.launch --nproc_per_node=8 --use_env SarcDetect.py \
--config ./configs/SarcDetect.yaml \
--output_dir output/SarcDetect \
--checkpoint ALBEF.pth\
 > ./sarc-detect.log