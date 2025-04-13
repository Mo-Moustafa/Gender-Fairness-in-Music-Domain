import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

def split_users(df, train_frac=0.6, val_frac=0.2, test_frac=0.2, seed=42):
    """
    Split users into train, validation, and test groups.
    For train users, all interactions will be used.
    For validation and test users, we later split their interactions.
    """
    users = df['user_id'].unique()
    np.random.seed(seed)
    np.random.shuffle(users)
    n = len(users)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    
    train_users = users[:n_train]
    val_users = users[n_train:n_train+n_val]
    test_users = users[n_train+n_val:]

    return train_users, val_users, test_users       # Returns back user_ids for each split


def holdout_interactions(df, holdout_frac=0.2, seed=42):
    """
    For each user, split their interactions into a training (80%) and holdout (20%) portion.
    Returns two DataFrames: one with the training interactions and one with the held-out interactions.
    """
    train_list = []
    holdout_list = []
    grouped = df.groupby('user_id')
    for user, group in grouped:
        # If a user has only one interaction, use it in training.
        if len(group) <= 1:
            train_list.append(group)
            continue
        user_train, user_holdout = train_test_split(group, test_size=holdout_frac, random_state=seed)
        train_list.append(user_train)
        holdout_list.append(user_holdout)
    train_df = pd.concat(train_list)
    holdout_df = pd.concat(holdout_list)
    return train_df, holdout_df         # Returns back two DataFrames, one for training and one for testing