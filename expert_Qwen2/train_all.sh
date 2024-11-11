python train.py \
--dataset funny \
--train_path ../urfunny_data/data_raw/test_data.json \
--val_path ../urfunny_data/data_raw/val_data.json \
--test_path ../urfunny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./0.5_qwen_urfunny_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset funny \
--train_path ../data_splits/urfunny_data_split_output/urfunny_R_dataset_train.json \
--val_path  ../data_splits/urfunny_data_split_output/urfunny_R_dataset_val.json \
--test_path  ../data_splits/urfunny_data_split_output/urfunny_R_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./0.5_qwen_urfunny_R_model \
--load_model_name ./0.5_qwen_urfunny_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset funny \
--train_path  ../data_splits/urfunny_data_split_output/urfunny_U_dataset_train.json \
--val_path  ../data_splits/urfunny_data_split_output/urfunny_U_dataset_val.json \
--test_path  ../data_splits/urfunny_data_split_output/urfunny_U_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./0.5_qwen_urfunny_U_model \
--load_model_name ./0.5_qwen_urfunny_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset funny \
--train_path  ../data_splits/urfunny_data_split_output/urfunny_AS_dataset_train.json \
--val_path  ../data_splits/urfunny_data_split_output/urfunny_AS_dataset_val.json \
--test_path  ../data_splits/urfunny_data_split_output/urfunny_AS_dataset_test.json \
--image_data_path ../funny_data/data_raw/images \
--save_path ./0.5_qwen_urfunny_AS_model \
--load_model_name ./0.5_qwen_urfunny_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;


python train.py \
--dataset mmsd \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset mmsd \
--train_path  ../data_splits/mmsd_data_split_output/mmsd_AS_dataset_train.json \
--val_path  ../data_splits/mmsd_data_split_output/mmsd_AS_dataset_val.json \
--test_path  ../data_splits/mmsd_data_split_output/mmsd_AS_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_AS_model \
--load_model_name ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset mmsd \
--train_path  ../data_splits/mmsd_data_split_output/mmsd_R_dataset_train.json \
--val_path  ../data_splits/mmsd_data_split_output/mmsd_R_dataset_val.json \
--test_path  ../data_splits/mmsd_data_split_output/mmsd_R_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_R_model \
--load_model_name ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset mmsd \
--train_path  ../data_splits/mmsd_data_split_output/mmsd_U_dataset_train.json \
--val_path  ../data_splits/mmsd_data_split_output/mmsd_U_dataset_val.json \
--test_path  ../data_splits/mmsd_data_split_output/mmsd_U_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--save_path ./0.5_qwen_mmsd_U_model \
--load_model_name ./0.5_qwen_mmsd_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset mustard \
--train_path  ../mustard_data/data_raw/mustard_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./0.5_qwen_mustard_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;


python train.py \
--dataset mustard \
--train_path  ../data_splits/mustard_data_split_output/mustard_AS_dataset_train.json \
--val_path  ../data_splits/mustard_data_split_output/mustard_AS_dataset_test.json \
--test_path  ../data_splits/mustard_data_split_output/mustard_AS_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./0.5_qwen_mustard_AS_model \
--load_model_name ./0.5_qwen_mustard_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 25 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset mustard \
--train_path  ../data_splits/mustard_data_split_output/mustard_R_dataset_train.json \
--val_path  ../data_splits/mustard_data_split_output/mustard_R_dataset_test.json \
--test_path  ../data_splits/mustard_data_split_output/mustard_R_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./0.5_qwen_mustard_R_model \
--load_model_name ./0.5_qwen_mustard_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--seed 32 \
--max_length 512;

python train.py \
--dataset mustard \
--train_path  ../data_splits/mustard_data_split_output/mustard_U_dataset_train.json \
--val_path  ../data_splits/mustard_data_split_output/mustard_U_dataset_test.json \
--test_path  ../data_splits/mustard_data_split_output/mustard_U_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./0.5_qwen_mustard_U_model \
--load_model_name ./0.5_qwen_mustard_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--device 0 \
--seed 32 \
--max_length 512;


