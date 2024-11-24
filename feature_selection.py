import numpy as np
import pandas as pd

def entropy(y):
    probs = [sum(y == c) / len(y) for c in set(y)]
    return sum(-p * np.log2(p) for p in probs)

def calculate_entropies(df, columns):
    entropies = []
    for column in columns:
        entropies.append(entropy(df[column]))
    return entropies

def drop_high_entropy_features(df, columns_to_drop):
    return df.drop(columns_to_drop, axis=1)
