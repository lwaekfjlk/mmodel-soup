python train.py \
--load_model_name Qwen/Qwen2-0.5B \
--dataset mustard \
--train_path ../mustard_data/data_raw/mustard_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./1.5_qwen_mustard_baseline_model \
--model_size 1.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--load_model_name Qwen/Qwen2-0.5B \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./1.5_qwen_mustard_AS_new_model \
--model_size 1.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--load_model_name Qwen/Qwen2-0.5B \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./1.5_qwen_mustard_R_new_model \
--model_size 1.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--load_model_name Qwen/Qwen2-0.5B \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./1.5_qwen_mustard_U_new_model \
--model_size 1.5 \
--batch_size 1 \
--eval_steps 10 \
--device 0 \
--max_length 512;