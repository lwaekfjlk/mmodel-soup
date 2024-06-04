#!/bin/bash

# Function to run training for combined datasets
run_training() {
  local combined_dataset_names=("$1" "$2" "$3")
  local combined_train_paths=("$4" "$5" "$6")
  local combined_val_paths=("$7" "$8" "$9")
  local combined_test_paths=("${10}" "${11}" "${12}")
  local combined_image_data_paths=("${13}" "${14}" "${15}")
  local combined_max_lengths=("${16}" "${17}" "${18}")
  local save_path_prefix="${19}"
  local batch_size="${20}"
  local eval_steps="${21}"
  local epochs="${22}"

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
EPOCHS=5

# Dataset-specific parameters
COMBINED_DATASETS=("irfl" "irfl" "irfl")

SARC_VAL_PATH="../irfl_data/data_raw/irfl_idiom_dataset_test.json"
SARC_TEST_PATH="../irfl_data/data_raw/irfl_idiom_dataset_test.json"
SARC_IMAGE_DATA_PATH="../irfl_data/data_raw/images"

MUSTARD_VAL_PATH="../irfl_data/data_raw/irfl_idiom_dataset_test.json"
MUSTARD_TEST_PATH="../irfl_data/data_raw/irfl_idiom_dataset_test.json"
MUSTARD_IMAGE_DATA_PATH="../irfl_data/data_raw/images"

VAL_PATHS=("$SARC_VAL_PATH" "$MUSTARD_VAL_PATH")
TEST_PATHS=("$SARC_TEST_PATH" "$MUSTARD_TEST_PATH")
IMAGE_DATA_PATHS=("$SARC_IMAGE_DATA_PATH" "$MUSTARD_IMAGE_DATA_PATH")

# Maximum lengths for tokenized sequences
MAX_LENGTHS=(512 512 512)

# Run AS experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../irfl_data/data_split_output/irfl_idiom_AS_dataset_train.json" "../irfl_data/data_split_output/irfl_metaphor_AS_dataset_train.json"  "../irfl_data/data_split_output/irfl_simile_AS_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_irfl_irfl_AS_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"

# Run R experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../irfl_data/data_split_output/irfl_idiom_R_dataset_train.json" "../irfl_data/data_split_output/irfl_metaphor_R_dataset_train.json" "../irfl_data/data_split_output/irfl_simile_R_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_irfl_irfl_R_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"

# Run U experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../irfl_data/data_split_output/irfl_idiom_U_dataset_train.json" "../irfl_data/data_split_output/irfl_metaphor_U_dataset_train.json" "../irfl_data/data_split_output/irfl_simile_U_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_irfl_irfl_U_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"

# Run baseline experiment
run_training "${COMBINED_DATASETS[@]}" \
  "../irfl_data/data_raw/irfl_dataset_train.json" "../irfl_data/data_raw/irfl_dataset_train.json" \
  "${VAL_PATHS[@]}" \
  "${TEST_PATHS[@]}" \
  "${IMAGE_DATA_PATHS[@]}" \
  "${MAX_LENGTHS[@]}" \
  "./blip2_irfl_irfl_baseline_model" \
  "$BATCH_SIZE" "$EVAL_STEPS" "$EPOCHS"
