export MASTER_ADDR=localhost
export MASTER_PORT=12355
CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch --nproc_per_node=4 --use_env IRFL.py \
--config ./configs/IRFL_logits.yaml \
--output_dir ./output/IRFL-logits \
--checkpoint /home/zqi2/mmodel-soup/expert_ALBEF/output/IRFL/checkpoint_best.pth \
--no-train
# > ./sarc-detect.log
# -m torch.distributed.launch --nproc_per_node=8 --use_env
# pip install torch torchvision torchaudio cudatoolkit=11.8 -c pytorch
# 