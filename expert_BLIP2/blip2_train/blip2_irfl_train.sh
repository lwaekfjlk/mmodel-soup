python train.py \
--dataset irfl \
--train_path ../../irfl_data/data_split_output/irfl_idiom_AS_dataset_train.json \
--val_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--test_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--image_data_path ../../irfl_data/data_raw/images \
--save_path ./blip2_irfl_idiom_AS_model;

python train.py \
--dataset irfl \
--train_path ../../irfl_data/data_split_output/irfl_idiom_R_dataset_train.json \
--val_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--test_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--image_data_path ../../irfl_data/data_raw/images \
--save_path ./blip2_irfl_idiom_R_model;

python train.py \
--dataset irfl \
--train_path ../../irfl_data/data_split_output/irfl_idiom_U_dataset_train.json \
--val_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--test_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--image_data_path ../../irfl_data/data_raw/images \
--save_path ./blip2_irfl_idiom_U_model;

python train.py \
--dataset irfl \
--train_path ../../irfl_data/data_raw/irfl_idiom_dataset_train.json \
--val_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--test_path ../../irfl_data/data_raw/irfl_idiom_dataset_test.json \
--image_data_path ../../irfl_data/data_raw/images \
--save_path ./blip2_irfl_baseline_model;