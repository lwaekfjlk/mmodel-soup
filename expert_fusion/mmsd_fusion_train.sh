python blip2_fusion_train_with_classifier.py \
--dataset mmsd \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./mmsd_blip2_fuser_with_classifier_A100_multi-task \
--batch_size 60 \
--eval_steps 300 \
--epochs 10 \
--max_length 512 \
--lr 2e-4 \
>mmsd_blip2_fusion_with_task_loss_A100_multi_task
