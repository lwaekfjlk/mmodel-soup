#python train.py \
#--dataset mustard \
#--train_path ../mustard_data/data_split_output/mustard_AS_dataset_train.json \
#--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
#--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
#--image_data_path ../mustard_data/data_raw/images \
#--save_path ./0.5_qwen_mustard_AS_model \
#--model_size 0.5 \
#--batch_size 1 \
#--eval_steps 25 \
#--epochs 5 \
#--device 2 \
#--max_length 512;

#python train.py \
#--dataset mustard \
#--train_path ../mustard_data/data_split_output/mustard_R_dataset_train.json \
#--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
#--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
#--image_data_path ../mustard_data/data_raw/images \
#--save_path ./0.5_qwen_mustard_R_model \
#--model_size 0.5 \
#--batch_size 1 \
#--eval_steps 10 \
#--epochs 5 \
#--device 2 \
#--max_length 512;

#python train_new.py \
#--dataset mustard \
#--train_path ../mustard_data/data_split_output/mustard_U_dataset_train.json \
#--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
#--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
#--image_data_path ../mustard_data/data_raw/images \
#--save_path ./0.5_qwen_mustard_U_model \
#--model_size 0.5 \
#--batch_size 1 \
#--eval_steps 10 \
#--device 2 \
#--max_length 512;


python train.py \
--dataset mustard \
--train_path ../mustard_data/data_raw/mustard_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./0.5_qwen_mustard_baseline_model \
--model_size 0.5 \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 2 \
--max_length 512;
