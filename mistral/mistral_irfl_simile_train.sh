python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_simile_AS_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_simile_AS_model \
--batch_size 2 \
--eval_steps 300 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_simile_R_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_simile_R_model \
--batch_size 2 \
--eval_steps 150 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_split_output/irfl_simile_U_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_simile_U_model \
--batch_size 2 \
--eval_steps 150 \
--max_length 512;

python train.py \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_simile_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_simile_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--save_path ./mistral_irfl_simile_baseline_model \
--batch_size 2 \
--eval_steps 150 \
--max_length 512;