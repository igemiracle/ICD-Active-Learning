import pandas as pd
import random

def load_initial_split(data_path='data/mimic3_data_test.pkl', seed=42, initial_size=100):
    df = pd.read_pickle(data_path).reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    labeled_set = df.iloc[:initial_size].copy()
    unlabeled_set = df.iloc[initial_size:].copy()

    return labeled_set, unlabeled_set