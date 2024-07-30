jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-test.log -err ./mmsd-test.err python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-r-test.log -err ./mmsd-r-test.err python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-u-test.log -err ./mmsd-u-test.err python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./mmsd-as-test.log -err ./mmsd-as-test.err  python train.py \
--mode test \
--test_dataset mmsd \
--test_path ../mmsd_data/data_raw/mmsd_dataset_test.json \
--image_data_path ../mmsd_data/data_raw/images \
--load_model_name blip2_mmsd_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512