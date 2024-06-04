python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;