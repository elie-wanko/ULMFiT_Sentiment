from glob import glob
from itertools import chain
from random import sample

from sklearn.model_selection import train_test_split


def create_splits(data, stratify_col):
    """
    Create the appropriate training(1:1:1) and validation sets(3:1:1)
    """
    
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data[stratify_col])
    
    # Model has to be trained on a set that has equal proportions of positive, neutral and negative tweets.
    pv_inds = train_data[train_data["airline_sentiment"]==2].index.tolist()
    nn_inds = train_data[train_data["airline_sentiment"]==1].index.tolist()
    nv_inds = train_data[train_data["airline_sentiment"]==0].index.tolist()
    nn_sample = sample(nn_inds, len(pv_inds))
    nv_sample = sample(nv_inds, len(pv_inds))
    train_data = train_data.loc[pv_inds + nn_sample + nv_sample]
    train_data
    
    # Validation set has to be representative of the real data in the ratio (3:1:1).
    pv_inds = val_data[val_data["airline_sentiment"]==2].index.tolist()
    nn_inds = val_data[val_data["airline_sentiment"]==1].index.tolist()
    nv_inds = val_data[val_data["airline_sentiment"]==0].index.tolist()
    nn_sample = sample(nn_inds, len(pv_inds))
    nv_sample = sample(nv_inds, 3*len(pv_inds))
    val_data = val_data.loc[pv_inds + nn_sample + nv_sample]
    
    return train_data, val_data
