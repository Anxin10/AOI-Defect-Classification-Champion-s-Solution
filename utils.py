import os
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import WeightedRandomSampler

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to True if input sizes are constant for speed

def get_folds(df, n_folds=5, seed=42):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Label'])):
        df.loc[val_idx, 'fold'] = fold
    return df

def get_balance_sampler(df):
    """
    Calculate sampling weights to balance class distribution.
    Rare classes (like Class 2) will be sampled more frequently.
    """
    # 1. Calculate count for each class
    class_counts = df['Label'].value_counts().sort_index().values
    
    # 2. Calculate weight for each class (inverse frequency)
    # Using 1.0 / count to give higher weight to rare classes
    class_weights = 1.0 / class_counts
    
    # 3. Assign weight to each sample
    sample_weights = [class_weights[label] for label in df['Label'].values]
    
    # 4. Create WeightedRandomSampler
    # replacement=True allows oversampling rare classes
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True
    )
    
    return sampler

def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')
