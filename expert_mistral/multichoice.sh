python train.py \
--dataset nycartoon \
--train_path ../nycartoon_data/data_raw/nycartoon_dataset_train_multichoice.json \
--val_path ../nycartoon_data/data_raw/nycartoon_dataset_test_multichoice.json \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test_multichoice.json \
--image_data_path ../nycartoon_data/data_raw/images \
--save_path ./mistral_nycartoon_AS_model \
--batch_size 1 \
--eval_steps 2000 \
--epochs 5 \
--answer_options 5 \
--max_length 512;

