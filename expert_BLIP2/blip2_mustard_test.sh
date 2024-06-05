python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset mustard \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_model_name blip2_mustard_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;