python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_AS_model;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_R_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_R_model;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_U_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_U_model;

python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_raw/nycartoon_dataset_train.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./blip2_nycartoon_baseline_model;