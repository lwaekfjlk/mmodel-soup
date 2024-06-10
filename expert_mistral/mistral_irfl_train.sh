python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_AS_dataset_train.json  \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_AS_model \
--batch_size 2 \
--eval_steps 300 \
--device 2 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_R_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_R_model \
--batch_size 2 \
--eval_steps 150 \
--device 2 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_U_dataset_train.json  \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_U_model \
--batch_size 2 \
--eval_steps 150 \
--device 2 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_dataset_train.json  \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_baseline_model \
--batch_size 2 \
--eval_steps 150 \
--device 2 \
--max_length 512;