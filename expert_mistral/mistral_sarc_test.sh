python train.py \
--mode test \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_AS_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name mistral_sarc_U_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;

python train.py \
--mode test \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_AS_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name mistral_sarc_AS_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;


python train.py \
--mode test \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_AS_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name mistral_sarc_R_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;


python train.py \
--mode test \
--dataset sarc \
--train_path ../sarc_data/data_split_output/sarc_AS_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name mistral_sarc_baseline_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--max_length 512;
