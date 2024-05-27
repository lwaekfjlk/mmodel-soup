python train.py \
--train_path ../../sarc_data/data_split_output/sarc_AS_dataset_train.json \
--val_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--test_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--image_data_path ../../sarc_data/data_raw/images \
--save_path ./blip2_sarc_AS_model;

python train.py \
--train_path ../../sarc_data/data_split_output/sarc_R_dataset_train.json \
--val_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--test_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--image_data_path ../../sarc_data/data_raw/images \
--save_path ./blip2_sarc_R_model;

python train.py \
--train_path ../../sarc_data/data_split_output/sarc_U_dataset_train.json \
--val_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--test_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--image_data_path ../../sarc_data/data_raw/images \
--save_path ./blip2_sarc_U_model;

python train.py \
--train_path ../../sarc_data/data_split_output/sarc_dataset_train.json \
--val_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--test_path ../../sarc_data/data_split_output/sarc_dataset_test.json \
--image_data_path ../../sarc_data/data_raw/images \
--save_path ./blip2_sarc_baseline_model;