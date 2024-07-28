python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;