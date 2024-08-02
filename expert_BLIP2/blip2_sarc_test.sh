jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-test.log -err ./sarc-test.err python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-r-test.log -err ./sarc-r-test.err python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-u-test.log -err ./sarc-u-test.err python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./sarc-as-test.log -err ./sarc-as-test.err  python train.py \
--mode test \
--test_dataset sarc \
--test_path ../sarc_data/data_raw/sarc_dataset_test.json \
--image_data_path ../sarc_data/data_raw/images \
--load_model_name blip2_sarc_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512;