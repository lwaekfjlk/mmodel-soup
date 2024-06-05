#python train.py \
#--dataset nycartoon \
#--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train_NEW.json \
#--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
#--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
#--image_data_path ../nycartoon_data/data_raw/images \
#--save_path ./mistral_nycartoon_AS_model \
#--batch_size 1 \
#--eval_steps 300 \
#--epochs 5 \
#--max_length 512;

#python train.py \
#--dataset nycartoon \
#--train_path ../nycartoon_data/data_split_output/nycartoon_R_dataset_train_NEW.json \
#--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
#--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
#--image_data_path ../nycartoon_data/data_raw/images \
#--save_path ./mistral_nycartoon_R_model \
#--batch_size 1 \
#--eval_steps 15000 \
#--epochs 5 \
#--max_length 512;

#python train.py \
#--dataset nycartoon \
#--train_path ../nycartoon_data/data_split_output/nycartoon_U_dataset_train_NEW.json \
#--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
#--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
#--image_data_path ../nycartoon_data/data_raw/images \
#--save_path ./mistral_nycartoon_U_model \
#--batch_size 1 \
#--eval_steps 15000 \
#--epochs 5 \
#--device 2 \
#--max_length 512;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_raw/nycartoon_dataset_train_NEW.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test_NEW.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./mistral_nycartoon_baseline_model \
--batch_size 3 \
--eval_steps 15000 \
--epochs 5 \
--device 2 \
--max_length 512;
