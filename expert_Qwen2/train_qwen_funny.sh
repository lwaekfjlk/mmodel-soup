python train.py \
--dataset funny \
--train_path /storage/mmodel-soup/new_data_splits/urfunny_data_split_output/urfunny_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../funny_data/data_raw/test_data.json \
--test_path ../funny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./7_qwen_funny_R_model \
--load_model_name ./7_qwen_funny_R_model \
--model_size 7 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--dataset funny \
--train_path /storage/mmodel-soup/new_data_splits/urfunny_data_split_output/urfunny_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../funny_data/data_raw/test_data.json \
--test_path ../funny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./7_qwen_funny_U_model \
--load_model_name ./7_qwen_funny_U_model \
--model_size 7 \
--batch_size 1 \
--eval_steps 10 \
--device 0 \
--max_length 512;

python train.py \
--dataset funny \
--train_path /storage/mmodel-soup/new_data_splits/urfunny_data_split_output/urfunny_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../funny_data/data_raw/test_data.json \
--test_path ../funny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./7_qwen_funny_AS_model \
--load_model_name ./7_qwen_funny_AS_model \
--model_size 7 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--max_length 512;

