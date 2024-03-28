export CONDA_VISIBLE_DEVICES=7

python hf_image_cls_multiproc.py --from_pretrained THUDM/cogagent-chat-hf --bf16 --image_dir ../sarc_data/image_data --save_file ../sarc_data/sarc_image_cls.txt --query "Think step by step. Does this image contain very obvious sarcasm? Answer yes or no first. Then explain the reason. " --num_processes 1
