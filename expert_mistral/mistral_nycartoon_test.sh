python train.py \
--mode test \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train_NEW.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name ./mistral_nycartoon_AS_model \
--batch_size 3 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--answer_options 5 \
--max_length 512;

python train.py \
--mode test \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train_NEW.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name ./mistral_nycartoon_R_model \
--batch_size 3 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--answer_options 5 \
--max_length 512;


python train.py \
--mode test \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train_NEW.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name ./mistral_nycartoon_U_model \
--batch_size 3 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--answer_options 5 \
--max_length 512;


python train.py \
--mode test \
--dataset nycartoon \
--train_path ../nycartoon_data/data_split_output/nycartoon_AS_dataset_train_NEW.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name ./mistral_nycartoon_baseline_model \
--batch_size 3 \
--eval_steps 10 \
--epochs 5 \
--device 0 \
--answer_options 5 \
--max_length 512;
