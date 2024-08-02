jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-as.log -err ./mustard-as.err  python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_split_output/mustard_AS_dataset_test_cogvlm2_qwen2.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_from_ckpt ./blip2_mustard_baseline_model \
--save_path ./blip2_mustard_AS_model \
--batch_size 5 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-r.log -err ./mustard-r.err python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_split_output/mustard_R_dataset_test_cogvlm2_qwen2.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_from_ckpt ./blip2_mustard_baseline_model \
--save_path ./blip2_mustard_R_model \
--batch_size 5 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard-u.log -err ./mustard-u.err python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_split_output/mustard_U_dataset_test_cogvlm2_qwen2.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_from_ckpt ./blip2_mustard_baseline_model \
--save_path ./blip2_mustard_U_model \
--batch_size 5 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mustard.log -err ./mustard.err python train.py \
--dataset mustard \
--train_path ../mustard_data/data_raw/mustard_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./blip2_mustard_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512