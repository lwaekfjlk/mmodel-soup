python -m torch.distributed.launch --nproc_per_node=8 --use_env BiCLS.py \
--config ./configs/BiCLS.yaml \
--output_dir output/BiCLS \
--checkpoint ALBEF.pth
#  > ./bi-cls.log