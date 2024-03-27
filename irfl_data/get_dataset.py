from datasets import Dataset
Dataset.cleanup_cache_files
from datasets import load_dataset
import pandas as pd

IRFL_images = load_dataset("lampent/IRFL", data_files='IRFL_images.zip')['train']

# IRFL dataset of figurative phrase-image pairs (10k+ images)
IRFL_idioms_dataset = load_dataset("lampent/IRFL", 'idioms-dataset')['dataset']
IRFL_similes_dataset = load_dataset("lampent/IRFL", 'similes-dataset')['dataset']
IRFL_metaphors_dataset = load_dataset("lampent/IRFL", 'metaphors-dataset')['dataset']

print('Successfully loaded IRFL dataset and tasks')

print(pd.DataFrame(IRFL_idioms_dataset).head())

pd.DataFrame(IRFL_metaphors_dataset).head()

pd.DataFrame(IRFL_similes_dataset).head()