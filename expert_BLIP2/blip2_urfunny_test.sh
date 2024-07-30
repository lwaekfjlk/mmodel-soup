jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-test.log -err ./urfunny-test.err python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_baseline_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-r-test.log -err ./urfunny-r-test.err python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_R_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-u-test.log -err ./urfunny-u-test.err python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_U_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512

jbsub -mem 80g -cores 20+1 -q alt_24h -require h100 -out ./urfunny-as-test.log -err ./urfunny-as-test.err  python train.py \
--mode test \
--test_dataset urfunny \
--test_path ../urfunny_data/data_raw/urfunny_dataset_test.json \
--image_data_path ../urfunny_data/data_raw/images \
--load_model_name blip2_urfunny_AS_model \
--batch_size 10 \
--eval_steps 10 \
--epochs 5 \
--max_length 512