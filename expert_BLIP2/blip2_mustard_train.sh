export SEED=3;
python train.py \
--dataset mustard \
--train_path ../mustard_data/data_raw/mustard_dataset_train.json \
--val_path ../mustard_data/data_raw/mustard_dataset_val.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--save_path ./blip2_mustard_baseline_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 10 \
--lr 4e-4 \
--seed ${SEED} \
--max_length 512

export SEED=3;
python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_AS_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_split_output/mustard_AS_dataset_val_cogvlm2_qwen2.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_from_ckpt ./blip2_mustard_baseline_model_${SEED} \
--save_path ./blip2_mustard_AS_model_${SEED} \
--batch_size 10 \
--eval_steps 100 \
--lr 4e-5 \
--epochs 10 \
--seed ${SEED} \
--max_length 512

export SEED=3;
python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_R_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_split_output/mustard_R_dataset_val_cogvlm2_qwen2.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_from_ckpt ./blip2_mustard_baseline_model_${SEED} \
--save_path ./blip2_mustard_R_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 100 \
--lr 1e-5 \
--seed ${SEED} \
--max_length 512

export SEED=3;
python train.py \
--dataset mustard \
--train_path ../mustard_data/data_split_output/mustard_U_dataset_train_cogvlm2_qwen2.json \
--val_path ../mustard_data/data_split_output/mustard_U_dataset_val_cogvlm2_qwen2.json \
--test_path ../mustard_data/data_raw/mustard_dataset_test.json \
--image_data_path ../mustard_data/data_raw/images \
--load_from_ckpt ./blip2_mustard_baseline_model_${SEED} \
--save_path ./blip2_mustard_U_model_${SEED} \
--batch_size 10 \
--eval_steps 10 \
--epochs 100 \
--lr 1e-5 \
--seed ${SEED} \
--max_length 512
