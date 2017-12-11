import torch
import math
import random
from collections import defaultdict


def generate_splits(indices, train_per=.8, valid_per=.1, shuffle=True):

    if isinstance(indices, int) and indices > 0:
        indices = torch.LongTensor([i for i in range(indices)])
        if shuffle:
            random.shuffle(indices)
    elif isinstance(indices, (list, tuple)):
        indices = torch.LongTensor(indices)
        if shuffle:
            random.shuffle(indices)

    elif isinstance(indices, torch.LongTensor):
        if shuffle:
            indices = indices.clone()
            random.shuffle(indices)
    else:
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if any(indices.lt(0)):
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if train_per <= 0 or train_per >= 1:
        raise Exception("train_per must be a float in [0, 1].")

    if valid_per < 0 or valid_per >= 1:
        raise Exception("valid_per must be a float in (0, 1].")

    test_per = 1 - (train_per + valid_per)

    data_size = indices.size(0)
    
    test_size = math.ceil(test_per * data_size)
    valid_size = math.ceil(valid_per * data_size)
    train_size = data_size - (test_size + valid_size)
    
    if train_size < 1:
        raise Exception("Not enough data points.")

    if valid_per > 0:
    
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:] 

        return train_indices, valid_indices, test_indices

    else:

        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        return train_indices, test_indices

def stratified_generate_splits(indices, labels, train_per=.8, valid_per=.1):
    if isinstance(indices, int) and indices > 0:
        # TODO rethink this what if this number is differnt than labels
        indices = torch.LongTensor([i for i in range(indices)])
    elif isinstance(indices, (list, tuple)):
        indices = torch.LongTensor(indices)
    elif isinstance(indices, torch.LongTensor):
        pass 
    else:
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if any(indices.lt(0)):
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if isinstance(labels, (list, tuple)):
        labels = torch.LongTensor(labels)
    elif isinstance(labels, torch.LongTensor):
        pass 
    else:
        raise Exception("labels must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if any(labels.lt(0)):
        raise Exception("labels must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if labels.size(0) != indices.size(0):
        raise Exception("labels and indices must be the same size.")

    label2indices = defaultdict(list)
    label2splits = {}

    for index, label in zip(indices, labels):
        label2indices[label].append(index)
    
    train_indices = []
    valid_indices = []
    test_indices = []

    labels = label2indices.keys()

    for label in labels:

        label_indices = label2indices[label]
        
        if valid_per > 0:
            
            tr_idx, val_idx, te_idx = generate_splits(
                label_indices, train_per=train_per, valid_per=valid_per)
            train_indices.append(tr_idx)
            valid_indices.append(val_idx)
            test_indices.append(te_idx)

        else:
            
            tr_idx, te_idx = generate_splits(
                label_indices, train_per=train_per, valid_per=valid_per)
            train_indices.append(tr_idx)
            test_indices.append(te_idx)

    train_indices = torch.cat(train_indices)
    random.shuffle(train_indices)

    test_indices = torch.cat(test_indices)
    random.shuffle(test_indices)

    if valid_per > 0:

        valid_indices = torch.cat(valid_indices)
        random.shuffle(valid_indices)
        
        return train_indices, valid_indices, test_indices

    else:

        return train_indices, test_indices

def kfold_iter(indices, num_folds, shuffle=True, valid_per=0):

    if isinstance(indices, (list, tuple)):
        indices = torch.LongTensor(indices)
        if shuffle:
            random.shuffle(indices)

    elif isinstance(indices, torch.LongTensor):
        if shuffle:
            indices = indices.clone()
            random.shuffle(indices)
    else:
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if any(indices.lt(0)):
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if not isinstance(num_folds, int) or num_folds < 2:
        raise Exception("num_folds must be positive integer greater than 1.")

    if valid_per < 0 or valid_per >= 1:
        raise Exception("valid_per must be float in range (0, 1].")

    fold_size = math.ceil(indices.size(0) / num_folds)
    real_num_folds = math.ceil(indices.size(0) / fold_size)

    if real_num_folds != num_folds:
        raise Exception(
            "Not enough data points for {} folds.".format(num_folds))

    if valid_per > 0:
        valid_size = math.ceil(valid_per * (indices.size(0) - fold_size))
    else:
        valid_size = 0

    for n_fold, i in enumerate(range(0, indices.size(0), fold_size), 1):
        test_indices = indices[i:i + fold_size]

        if n_fold == 1:
            train_valid_indices = indices[fold_size:]
        elif n_fold == num_folds:
            train_valid_indices = indices[:i]
        else:
            train_valid_indices = torch.cat([indices[:i], 
                                             indices[i + fold_size:]])
        if valid_size == 0:
            yield train_valid_indices, test_indices
        else:
            random.shuffle(train_valid_indices)
            train_indices = train_valid_indices[valid_size:]
            valid_indices = train_valid_indices[:valid_size]
            yield train_indices, valid_indices, test_indices

def stratified_kfold_iter(indices, labels, num_folds, valid_per=0):

    if isinstance(indices, (list, tuple)):
        indices = torch.LongTensor(indices)
    elif isinstance(indices, torch.LongTensor):
        pass 
    else:
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if any(indices.lt(0)):
        raise Exception("indices must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if isinstance(labels, (list, tuple)):
        labels = torch.LongTensor(labels)
    elif isinstance(labels, torch.LongTensor):
        pass 
    else:
        raise Exception("labels must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if any(labels.lt(0)):
        raise Exception("labels must be a list, tuple, or torch.LongTensor " \
                        "of nonnegative integers.")

    if labels.size(0) != indices.size(0):
        raise Exception("labels and indices must be the same size.")

    label2indices = defaultdict(list)
    label2iter = {}

    for index, label in zip(indices, labels):
        label2indices[label].append(index)
    
    labels = label2indices.keys()
    for label in labels:
        label_indices = label2indices[label]
        label2iter[label] = kfold_iter(
            label_indices, num_folds, valid_per=valid_per)

    for k in range(num_folds):
        train_indices = []
        valid_indices = []
        test_indices = []

        for label in labels:
            if valid_per > 0:
                tr, val, te = next(label2iter[label])
                train_indices.append(tr)
                valid_indices.append(val)
                test_indices.append(te)
            else:
                tr, te = next(label2iter[label])
                train_indices.append(tr)
                test_indices.append(te)
        
        train_indices = torch.cat(train_indices)
        random.shuffle(train_indices)

        test_indices = torch.cat(test_indices)
        random.shuffle(test_indices)

        if valid_per > 0:

            valid_indices = torch.cat(valid_indices)
            random.shuffle(valid_indices)
            
            yield train_indices, valid_indices, test_indices

        else:

            yield train_indices, test_indices
