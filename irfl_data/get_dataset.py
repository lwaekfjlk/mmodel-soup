from datasets import Dataset
Dataset.cleanup_cache_files
from datasets import load_dataset
import pandas as pd

# IRFL dataset of figurative phrase-image pairs (10k+ images)
IRFL_idioms_dataset = load_dataset("lampent/IRFL", 'idioms-dataset')['dataset']
IRFL_similes_dataset = load_dataset("lampent/IRFL", 'similes-dataset')['dataset']
IRFL_metaphors_dataset = load_dataset("lampent/IRFL", 'metaphors-dataset')['dataset']

print('Successfully loaded IRFL dataset and tasks')

idioms_df = pd.DataFrame(IRFL_idioms_dataset)
similes_df = pd.DataFrame(IRFL_similes_dataset)
metaphors_df = pd.DataFrame(IRFL_metaphors_dataset)

idioms_df_filtered = idioms_df[idioms_df['category'].notna()]
similes_df_filtered = similes_df[similes_df['category'].notna()]
metaphors_df_filtered = metaphors_df[metaphors_df['category'].notna()]

# take columns of 'uuid', 'phrase', 'category'
idioms_df_filtered = idioms_df_filtered[['uuid', 'phrase', 'category']]
similes_df_filtered = similes_df_filtered[['uuid', 'phrase', 'category']]
metaphors_df_filtered = metaphors_df_filtered[['uuid', 'phrase', 'category']]

# join three datasets
IRFL_dataset = pd.concat([idioms_df_filtered, similes_df_filtered, metaphors_df_filtered])

# if category is not "Figurative", change it to "Not Figurative"
IRFL_dataset['category'] = IRFL_dataset['category'].apply(lambda x: x if x == 'Figurative' else 'Not Figurative')
print("count of 'Figurative' and 'Not Figurative' categories: ", IRFL_dataset['category'].value_counts())

# shuffle the dataset and split into train, valid, and test
IRFL_dataset = IRFL_dataset.sample(frac=1).reset_index(drop=True)
IRFL_train = IRFL_dataset[:int(len(IRFL_dataset) * 0.84)]
IRFL_valid = IRFL_dataset[int(len(IRFL_dataset) * 0.84):int(len(IRFL_dataset) * 0.92)]
IRFL_test = IRFL_dataset[int(len(IRFL_dataset) * 0.92):]

# save the dataset
IRFL_train.to_csv('intermediate_data/train.csv', index=False)
IRFL_valid.to_csv('intermediate_data/valid.csv', index=False)
IRFL_test.to_csv('intermediate_data/test.csv', index=False)

print('Successfully saved IRFL dataset')