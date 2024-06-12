python train.py \
--mode test \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--load_model_name mistral_irfl_baseline_model \
--save_path ./mistral_mustard_U_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 3 \
--max_length 512;

python train.py \
--mode test \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--load_model_name mistral_irfl_AS_model \
--save_path ./mistral_mustard_AS_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 3 \
--max_length 512;


python train.py \
--mode test \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--load_model_name mistral_irfl_R_model \
--save_path ./mistral_mustard_R_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 3 \
--max_length 512;


python train.py \
--mode test \
--dataset irfl \
--train_path ../irfl_data/data_raw/irfl_dataset_train.json \
--val_path ../irfl_data/data_raw/irfl_dataset_test.json \
--test_path ../irfl_data/data_raw/irfl_dataset_test.json \
--image_data_path ../irfl_data/data_raw/images \
--load_model_name mistral_irfl_U_model \
--save_path ./mistral_mustard_U_model \
--batch_size 1 \
--eval_steps 10 \
--epochs 5 \
--device 3 \
--max_length 512;
