python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_AS_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./mistral_mustard_AS_model \
--batch_size 2 \
--eval_steps 25 \
--epochs 5 \
--device 1 \
--max_length 512;

python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_R_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./mistralnci_mustard_R_model \
--batch_size 2 \
--eval_steps 10 \
--epochs 5 \
--device 1 \
--max_length 512;

python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_U_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./mistral_mustard_U_model \
--batch_size 2 \
--eval_steps 10 \
--device 1 \
--max_length 512;


python train.py \
--dataset mustard \
--train_path ../mustard_data/data_raw/mustard_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_test.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./mistral_mustard_baseline_model \
--batch_size 2 \
--eval_steps 10 \
--epochs 5 \
--device 1 \
--max_length 512;
