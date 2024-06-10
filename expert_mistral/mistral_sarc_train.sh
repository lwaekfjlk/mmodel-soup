#python train.py \
#--dataset sarc \
#--train_path ../sarc_data/data_split_output/sarc_AS_dataset_train.json \
#--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
#--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
#--image_data_path ../sarc_data/data_raw/images \
#--save_path ./mistral_sarc_AS_model \
#--batch_size 1 \
#--eval_steps 500 \
#--epochs 5 \
#--max_length 512;

#python train.py \
#--dataset sarc \
#--train_path ../sarc_data/data_split_output/sarc_R_dataset_train.json \
#--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
#--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
#--image_data_path ../sarc_data/data_raw/images \
#--save_path ./mistral_sarc_R_model \
#--batch_size 1 \
#--eval_steps 500 \
#--epochs 5 \
#--max_length 512;

#python train.py \
#--dataset sarc \
#--train_path ../sarc_data/data_split_output/sarc_U_dataset_train.json \
#--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
#--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
#--image_data_path ../sarc_data/data_raw/images \
#--save_path ./mistral_sarc_U_model \
#--batch_size 1 \
#--eval_steps 4000 \
#--epochs 5 \
#--max_length 512;

python train.py \
--dataset sarc \
--train_path ../sarc_data/data_raw/sarc_dataset_train.json \
--val_path ../sarc_data/data_raw/sarc_dataset_test.json \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--save_path ./mistral_sarc_baseline_model \
--batch_size 1 \
--eval_steps 8000 \
--epochs 5 \
--device 3 \
--max_length 512;
