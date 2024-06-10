#!/bin/bash

# Function to run training for combined datasets
run_training() {
  local combined_dataset_names=("$1" "$2" "$3" "$4")
  local combined_train_paths=("$5" "$6" "$7" "$8")
  local combined_val_paths=("$9" "${10}" "${11}" "${12}")
  local combined_test_paths=("${13}" "${14}" "${15}" "${16}")
  local combined_image_data_paths=("${17}" "${18}" "${19}" "${20}")
  local combined_max_lengths=("${21}" "${22}" "${23}" "${24}")
  local save_path_prefix="${25}"
  local batch_size="${26}"
  local eval_steps="${27}"
  local epochs="${28}"

  python train.py \
    --dataset combined \
    --combined_dataset_names "${combined_dataset_names[@]}" \
    --combined_train_paths "${combined_train_paths[@]}" \
    --combined_val_paths "${combined_val_paths[@]}" \
    --combined_test_paths "${combined_test_paths[@]}" \
    --combined_image_data_paths "${combined_image_data_paths[@]}" \
    --combined_max_lengths "${combined_max_lengths[@]}" \
    --save_path "${save_path_prefix}" \
    --batch_size "${batch_size}" \
    --eval_steps "${eval_steps}" \
    --epochs "${epochs}"
}

# Common parameters
BATCH_SIZE=10
EVAL_STEPS=200
EPOCHS=2

# Dataset-specific parameters
COMBINED_DATASETS=("sarc" "mustard" "nycartoon" "irfl")

SARC_VAL_PATH="../sarc_data/data_raw/sarc_dataset_val.json"
SARC_TEST_PATH="../sarc_data/data_raw/sarc_dataset_test.json"
SARC_IMAGE_DATA_PATH="../sarc_data/data_raw/images"

MUSTARD_VAL_PATH="../mustard_data/data_raw/mustard_dataset_test.json"
MUSTARD_TEST_PATH="../mustard_data/data_raw/mustard_dataset_test.json"
MUSTARD_IMAGE_DATA_PATH="../mustard_data/data_raw/images"

NYCARTOON_VAL_PATH="../nycartoon_data/data_raw/nycartoon_dataset_val.json"
NYCARTOON_TEST_PATH="../nycartoon_data/data_raw/nycartoon_dataset_test.json"
NYCARTOON_IMAGE_DATA_PATH="../nycartoon_data/data_raw/images"

IRFL_VAL_PATH="../irfl_data/data_raw/irfl_dataset_val.json"
IRFL_TEST_PATH="../irfl_data/data_raw/irfl_dataset_test.json"
IRFL_IMAGE_DATA_PATH="../irfl_data/data_raw/images"

VAL_PATHS=("$SARC_VAL_PATH" "$MUSTARD_VAL_PATH" "$NYCARTOON_VAL_PATH" "$IRFL_VAL_PATH")
TEST_PATHS=("$SARC_TEST_PATH" "$MUSTARD_TEST_PATH" "$NYCARTOON_TEST_PATH" "$IRFL_TEST_PATH")
IMAGE_DATA_PATHS=("$SARC_IMAGE_DATA_PATH" "$MUSTARD_IMAGE_DATA_PATH" "$NYCARTOON_IMAGE_DATA_PATH" "$IRFL_IMAGE_DATA_PATH")

# Maximum lengths for tokenized sequences
MAX_LENGTHS=(512 512 512 512)

# Run AS experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../sarc_data/data_split_output/sarc_AS_dataset_train.json" "../mustard_data/data_split_output/mustard_AS_dataset_train.json" "../nycartoon_data/data_split_output/nycartoon_AS_dataset_train.json" "../irfl_data/data_split_output/irfl_AS_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_sarc_mustard_nycartoon_irfl_AS_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"

# Run R experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../sarc_data/data_split_output/sarc_R_dataset_train.json" "../mustard_data/data_split_output/mustard_R_dataset_train.json" "../nycartoon_data/data_split_output/nycartoon_R_dataset_train.json" "../irfl_data/data_split_output/irfl_R_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_sarc_mustard_nycartoon_irfl_R_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"

# Run U experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../sarc_data/data_split_output/sarc_U_dataset_train.json" "../mustard_data/data_split_output/mustard_U_dataset_train.json" "../nycartoon_data/data_split_output/nycartoon_U_dataset_train.json" "../irfl_data/data_split_output/irfl_U_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_sarc_mustard_nycartoon_irfl_U_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"

# Run baseline experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../sarc_data/data_raw/sarc_dataset_train.json" "../mustard_data/data_raw/mustard_dataset_train.json" "../nycartoon_data/data_raw/nycartoon_dataset_train.json" "../irfl_data/data_raw/irfl_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_sarc_mustard_nycartoon_irfl_baseline_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"
