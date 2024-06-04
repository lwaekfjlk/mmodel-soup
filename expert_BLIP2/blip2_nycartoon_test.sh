python train.py \
--mode test \
--test_dataset nycartoon \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name blip2_nycartoon_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset nycartoon \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name blip2_nycartoon_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset nycartoon \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name blip2_nycartoon_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset nycartoon \
--test_path ../nycartoon_data/data_raw/nycartoon_dataset_test.json \
--image_data_path ../nycartoon_data/data_raw/images \
--load_model_name blip2_nycartoon_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;