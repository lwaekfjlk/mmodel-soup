python train.py \
--mode test \
--dataset funny \
--val_path ../funny_data/data_raw/funny_dataset_test.json \
--test_path ../funny_data/data_raw/funny_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 7_qwen_funny_U_model \
--save_path ./qwen_funny_test_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;

python train.py \
--mode test \
--dataset funny \
--val_path ../funny_data/data_raw/funny_dataset_test.json \
--test_path ../funny_data/data_raw/funny_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 7_qwen_funny_AS_model \
--save_path ./qwen_funny_test_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;

python train.py \
--mode test \
--dataset funny \
--val_path ../funny_data/data_raw/funny_dataset_test.json \
--test_path ../funny_data/data_raw/funny_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 7_qwen_funny_R_model \
--save_path ./qwen_funny_test_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;

python train.py \
--mode test \
--dataset funny \
--val_path ../funny_data/data_raw/funny_dataset_test.json \
--test_path ../funny_data/data_raw/funny_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 7_qwen_funny_baseline_model \
--save_path ./qwen_funny_U_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;
