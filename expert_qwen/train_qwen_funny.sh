python train.py \
--dataset funny \
--train_path ../funny_data/data_raw/train_data.json \
--val_path ../funny_data/data_raw/val_data.json \
--test_path ../funny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./7_qwen_funny_AS_model \
--load_model_name ./7_qwen_funny_AS_model \
--model_size 7 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 2 \
--max_length 512;

# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_split_output/funny_R_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_R_model \
# --load_model_name ./7_qwen_funny_R_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 10 \
# --epochs 5 \
# --device 2 \
# --max_length 512;

# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_split_output/funny_U_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_U_model \
# --load_model_name ./7_qwen_funny_U_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 10 \
# --device 2 \
# --max_length 512;


# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_raw/funny_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_baseline_model \
# --load_model_name ./7_qwen_funny_baseline_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 10 \
# --epochs 5 \
# --device 2 \
# --max_length 512;


# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_split_output/funny_AS_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_AS_model \
# --load_model_name ./7_qwen_funny_AS_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 25 \
# --epochs 5 \
# --device 2 \
# --max_length 512;

# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_split_output/funny_R_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_R_model \
# --load_model_name ./7_qwen_funny_R_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 10 \
# --epochs 5 \
# --device 2 \
# --max_length 512;

# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_split_output/funny_U_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_U_model \
# --load_model_name ./7_qwen_funny_U_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 10 \
# --device 2 \
# --max_length 512;


# python train.py \
# --dataset funny \
# --train_path ../funny_data/data_raw/funny_dataset_train.json \
# --val_path ../funny_data/data_raw/funny_dataset_test.json \
# --test_path ../funny_data/data_raw/funny_dataset_test.json \
# --image_data_path ../funny_data/data_raw/images \
# --save_path ./7_qwen_funny_baseline_model \
# --load_model_name ./7_qwen_funny_baseline_model \
# --model_size 7 \
# --batch_size 1 \
# --eval_steps 10 \
# --epochs 5 \
# --device 2 \
# --max_length 512;
