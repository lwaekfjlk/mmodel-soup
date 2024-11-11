python train.py \
--mode test \
--dataset funny \
--train_path ../urfunny_data/data_raw/test_data.json \
--val_path ../urfunny_data/data_raw/test_data.json \
--test_path ../urfunny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 0.5_qwen_funny_baseline_model \
--save_path 0.5_qwen_funny_baseline_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--mode test \
--dataset funny \
--train_path ../urfunny_data/data_raw/test_data.json \
--val_path ../urfunny_data/data_raw/test_data.json \
--test_path ../urfunny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 0.5_qwen_funny_AS_model \
--save_path 0.5_qwen_funny_AS_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--mode test \
--dataset funny \
--train_path ../urfunny_data/data_raw/test_data.json \
--val_path ../urfunny_data/data_raw/test_data.json \
--test_path ../urfunny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 0.5_qwen_funny_R_model \
--save_path 0.5_qwen_funny_R_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--mode test \
--dataset funny \
--train_path ../urfunny_data/data_raw/test_data.json \
--val_path ../urfunny_data/data_raw/test_data.json \
--test_path ../urfunny_data/data_raw/test_data.json \
--image_data_path ../funny_data/data_raw/images \
--load_model_name 0.5_qwen_funny_U_model \
--save_path 0.5_qwen_funny_U_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;



python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_AS_model \
--save_path ./0.5_qwen_mmsd_AS_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;

python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_R_model \
--save_path ./0.5_qwen_mmsd_R_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;


python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_U_model \
--save_path ./0.5_qwen_mmsd_U_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;


python train.py \
--dataset mmsd \
--mode test \
--train_path ../mmsd_data/data_raw/mmsd_dataset_train.json \
--val_path ../mmsd_data/data_raw/mmsd_dataset_val.json \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name 0.5_qwen_mmsd_baseline_model \
--save_path ./0.5_qwen_mmsd_baseline_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 0 \
--test_batch_size 32 \
--max_length 512;


python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name "0.5_qwen_mustard_baseline_model" \
--save_path "0.5_qwen_mustard_baseline_model" \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;


python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name "0.5_qwen_mustard_AS_model" \
--save_path "0.5_qwen_mustard_AS_model" \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;


python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name "0.5_qwen_mustard_R_model" \
--save_path "0.5_qwen_mustard_R_model" \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;


python train.py \
--mode test \
--dataset mustard \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name "0.5_qwen_mustard_U_model" \
--save_path "0.5_qwen_mustard_U_model" \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;
