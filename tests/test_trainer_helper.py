import unittest
import nt.trainer.trainer_helper as trainer_helper
import torch
import math
import random
from collections import defaultdict

class TestTrainerHelper(unittest.TestCase):

    def test_stratified_generate_splits_exceptions(self):
        '''
        Test stratified_generate_splits throws the correct exceptions for bad 
        inputs.
        '''

        # Case 1: Different lengths of indices and labels inputs.
        data_size1 = 4
        indices1 = [i for i in range(data_size1)]
        labels1 = [0, 0, 1, 1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_generate_splits(indices1, labels1)
            self.assertEqual(
                "labels and indices must be the same size.",
                context.exception)
            
        # Case 2: indices is of some other type that is not a list, tuple, 
        # or a torch.LongTensor. For example, generators, while iterables,
        # would possibly break behavior.
        data_size2 = 4
        indices2 = set([i for i in range(0, data_size2)])
        labels2 = [0, 0, 1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_generate_splits(indices2, labels2)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 3: indices contains a negative integer. 
        # This wouldn't technically break the behavior but does not make sense
        # semantically if indices is supposed to correspond to dataset items.
        data_size3 = 4
        indices3 = [i for i in range(-1, data_size2 - 1)]
        labels3 = [0, 0, 1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_generate_splits(indices3, labels3)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 4: labels is of some other type that is not a list, tuple, 
        # or a torch.LongTensor. For example, generators, while iterables,
        # would possibly break behavior.
        data_size4 = 4
        indices4 = [i for i in range(0, data_size2)]
        labels4 = set([0, 0, 1, 1])
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_generate_splits(indices4, labels4)
            self.assertEqual(
                "labels must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 5: labels contains a negative integer. 
        # This wouldn't technically break the behavior but does not make sense
        # semantically if indices is supposed to correspond to dataset items.
        data_size5 = 4
        indices5 = [i for i in range(data_size2)]
        labels5 = [0, 0, -1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_generate_splits(indices5, labels5)
            self.assertEqual(
                "labels must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

    def test_stratified_generate_splits(self):
        '''
        Test stratified_generate_splits for cases with and without a 
        validation set.
        '''

        # Case 1: Without validation set.
        train_per1 = .5
        valid_per1 = 0
        data_size1 = 4 
        indices1 = [i for i in range(data_size1)]
        labels1 = [0, 0, 1, 1]
        
        tr_idx, te_idx = trainer_helper.stratified_generate_splits(
            indices1, labels1, train_per=train_per1, valid_per=valid_per1)

        train_label_counts1 = defaultdict(int)
        for idx in tr_idx:
            train_label_counts1[labels1[idx]] += 1
        for label, count in train_label_counts1.items():
            self.assertTrue(count == 1)
        
        test_label_counts1 = defaultdict(int)
        for idx in te_idx:
            test_label_counts1[labels1[idx]] += 1
        for label, count in test_label_counts1.items():
            self.assertTrue(count == 1)

        # Case 2: With validation set.
        train_per2 = .8
        valid_per2 = .1
        data_size2 = 20
        indices2 = [i for i in range(data_size2)]
        labels2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        
        tr_idx, val_idx, te_idx = trainer_helper.stratified_generate_splits(
            indices2, labels2, train_per=train_per2, valid_per=valid_per2)

        train_label_counts2 = defaultdict(int)
        for idx in tr_idx:
            train_label_counts2[labels2[idx]] += 1
        for label, count in train_label_counts2.items():
            self.assertTrue(count == 8)

        valid_label_counts2 = defaultdict(int)
        for idx in val_idx:
            valid_label_counts2[labels2[idx]] += 1
        for label, count in valid_label_counts2.items():
            self.assertTrue(count == 1)

        test_label_counts2 = defaultdict(int)
        for idx in te_idx:
            test_label_counts2[labels2[idx]] += 1
        for label, count in test_label_counts2.items():
            self.assertTrue(count == 1)

    def test_generate_splits_exceptions(self):
        '''
        Test that generate_splits throws the expected exceptions for bad 
        inputs.
        '''

        # Case 1: indices contains a negative integer. 
        # This wouldn't technically break the behavior but does not make sense
        # semantically if indices is supposed to correspond to dataset items.
        data_size1 = 10
        indices1 = [i for i in range(-1, data_size1 - 1)]
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(indices1)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)
 
        # Case 2: indices is of some other type that is not a list, tuple, 
        # or a torch.LongTensor. For example, generators, while iterables,
        # would possibly break behavior.
        data_size2 = 10
        indices2 = set([i for i in range(0, data_size2)])
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(indices2)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 3: Not enough data points.
        data_size3 = 2 
        indices3 = set([i for i in range(0, data_size3)])
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(indices3)
            self.assertEqual(
                'Not enough data points.',
                context.exception)

        # Case 4: train_per is not a float in [0, 1].
        train_per4 = 0
        data_size4 = 10
        indices4 = set([i for i in range(0, data_size4)])
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(
                indices4, train_per=train_per4)
            self.assertEqual(
                "train_per must be float in [0, 1]",
                context.exception)

        # Case 5: train_per is not a float in [0, 1].
        train_per5 = 1
        data_size5 = 10
        indices5 = set([i for i in range(0, data_size5)])
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(
                indices5, train_per=train_per5)
            self.assertEqual(
                "train_per must be float in [0, 1]",
                context.exception)

        # Case 6: valid_per is not a float in (0, 1].
        valid_per6 = -.1
        data_size6 = 10
        indices6 = set([i for i in range(0, data_size6)])
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(
                indices6, valid_per=valid_per6)
            self.assertEqual(
                "valid_per must be float in (0, 1]",
                context.exception)

        # Case 7: valid_per is not a float in (0, 1].
        valid_per7 = 1
        data_size7 = 10
        indices7 = set([i for i in range(0, data_size7)])
        with self.assertRaises(Exception) as context:
            trainer_helper.generate_splits(
                indices7, valid_per=valid_per7)
            self.assertEqual(
                "valid_per must be float in (0, 1]",
                context.exception)

    def test_generate_splits_shuffle(self):
        '''
        Test generate_splits respects shuffle option.
        '''

        # Avoid very rare case where shuffle result is the original order.
        random.seed(83483495)

        # Case 1: No shuffle, train and test only.
        train_per1 = .8
        valid_per1 = 0
        data_size1 = 10
        exp_train_size1 = math.ceil(data_size1 * train_per1)
        exp_test_size1 = math.ceil(
            data_size1 * (1 - (train_per1 + valid_per1)))
        shuffle1 = False
        indices1 = [i for i in range(data_size1)]

        tr_idx1, te_idx1 = trainer_helper.generate_splits(
            indices1, train_per=train_per1, valid_per=valid_per1, 
            shuffle=shuffle1)
        all_indices1 = tuple(tr_idx1.tolist() + te_idx1.tolist())
        self.assertTrue(all_indices1 == tuple(indices1))

        # Case 2: Shuffle, train and test only.
        train_per2 = .8
        valid_per2 = 0
        data_size2 = 10
        exp_train_size2 = math.ceil(data_size2 * train_per2)
        exp_test_size2 = math.ceil(
            data_size2 * (1 - (train_per2 + valid_per2)))
        shuffle2 = True
        indices2 = [i for i in range(data_size2)]

        tr_idx2, te_idx2 = trainer_helper.generate_splits(
            indices2, train_per=train_per2, valid_per=valid_per2, 
            shuffle=shuffle2)
        all_indices2 = tuple(tr_idx2.tolist() + te_idx2.tolist())
        self.assertTrue(all_indices2 != tuple(indices2))

        # Case 3: No shuffle, train, val, and test.
        train_per3 = .8
        valid_per3 = .1
        data_size3 = 10
        exp_train_size3 = math.ceil(data_size3 * train_per3)
        exp_valid_size3 = math.ceil(data_size3 * valid_per3)
        exp_test_size3 = math.ceil(
            data_size3 * (1 - (train_per3 + valid_per3)))
        shuffle3 = False
        indices3 = [i for i in range(data_size3)]

        tr_idx3, val_idx3, te_idx3 = trainer_helper.generate_splits(
            indices3, train_per=train_per3, valid_per=valid_per3, 
            shuffle=shuffle3)
        all_indices3 = tuple(
            tr_idx3.tolist() + val_idx3.tolist() + te_idx3.tolist())
        self.assertTrue(all_indices3 == tuple(indices3))

        # Case 4: Shuffle, train, val, and test.
        train_per4 = .8
        valid_per4 = .1
        data_size4 = 10
        exp_train_size4 = math.ceil(data_size4 * train_per4)
        exp_valid_size4 = math.ceil(data_size4 * valid_per4)
        exp_test_size4 = math.ceil(
            data_size4 * (1 - (train_per4 + valid_per4)))
        shuffle4 = True
        indices4 = [i for i in range(data_size4)]

        tr_idx4, val_idx4, te_idx4 = trainer_helper.generate_splits(
            indices4, train_per=train_per4, valid_per=valid_per4, 
            shuffle=shuffle4)
        all_indices4 = tuple(
            tr_idx4.tolist() + val_idx4.tolist() + te_idx4.tolist())
        self.assertTrue(all_indices4 != tuple(indices4))

    def test_generate_splits_no_validation_set(self):
        '''
        Test generate_splits when returning training and testing splits.
        '''

        # Case 1: list.
        train_per1 = .8
        valid_per1 = 0
        data_size1 = 10
        exp_train_size1 = math.ceil(data_size1 * train_per1)
        exp_test_size1 = math.ceil(
            data_size1 * (1 - (train_per1 + valid_per1)))
        shuffle1 = True
        indices1 = [i for i in range(data_size1)]

        tr_idx1, te_idx1 = trainer_helper.generate_splits(
            indices1, train_per=train_per1, valid_per=valid_per1, 
            shuffle=shuffle1)

        self.assertTrue(tr_idx1.size(0) == exp_train_size1)
        self.assertTrue(te_idx1.size(0) == exp_test_size1)

        all_indices1 = tr_idx1.tolist() + te_idx1.tolist()
        all_indices1.sort()
        self.assertTrue(tuple(all_indices1) == tuple(indices1))

        # Case 2: tuple.
        train_per2 = .8
        valid_per2 = 0
        data_size2 = 10
        exp_train_size2 = math.ceil(data_size2 * train_per2)
        exp_test_size2 = math.ceil(
            data_size2 * (1 - (train_per2 + valid_per2)))
        shuffle2 = True
        indices2 = tuple([i for i in range(data_size2)])

        tr_idx2, te_idx2 = trainer_helper.generate_splits(
            indices2, train_per=train_per2, valid_per=valid_per2, 
            shuffle=shuffle2)

        self.assertTrue(tr_idx2.size(0) == exp_train_size2)
        self.assertTrue(te_idx2.size(0) == exp_test_size2)

        all_indices2 = tr_idx2.tolist() + te_idx2.tolist()
        all_indices2.sort()
        self.assertTrue(tuple(all_indices2) == indices2)

        # Case 3: torch.LongTensor.
        train_per3 = .8
        valid_per3 = 0
        data_size3 = 10
        exp_train_size3 = math.ceil(data_size3 * train_per3)
        exp_test_size3 = math.ceil(
            data_size3 * (1 - (train_per3 + valid_per3)))
        shuffle3 = True
        indices3 = torch.arange(0, data_size3).long()

        tr_idx3, te_idx3 = trainer_helper.generate_splits(
            indices3, train_per=train_per3, valid_per=valid_per3, 
            shuffle=shuffle3)

        self.assertTrue(tr_idx3.size(0) == exp_train_size3)
        self.assertTrue(te_idx3.size(0) == exp_test_size3)

        all_indices3 = tr_idx3.tolist() + te_idx3.tolist()
        all_indices3.sort()
        self.assertTrue(tuple(all_indices3) == tuple(indices3.tolist()))

    def test_generate_splits_with_validation_set(self):
        '''
        Test generate_splits when returning training, validation, and testing
        splits.
        '''

        # Case 1: list.
        train_per1 = .8
        valid_per1 = .1
        data_size1 = 10
        exp_train_size1 = math.ceil(data_size1 * train_per1)
        exp_valid_size1 = math.ceil(data_size1 * valid_per1)
        exp_test_size1 = math.ceil(
            data_size1 * (1 - (train_per1 + valid_per1)))
        shuffle1 = True
        indices1 = [i for i in range(data_size1)]

        tr_idx1, val_idx1, te_idx1 = trainer_helper.generate_splits(
            indices1, train_per=train_per1, valid_per=valid_per1, 
            shuffle=shuffle1)

        self.assertTrue(tr_idx1.size(0) == exp_train_size1)
        self.assertTrue(val_idx1.size(0) == exp_valid_size1)
        self.assertTrue(te_idx1.size(0) == exp_test_size1)

        all_indices1 = tr_idx1.tolist() + val_idx1.tolist() + te_idx1.tolist()
        all_indices1.sort()
        self.assertTrue(tuple(all_indices1) == tuple(indices1))

        # Case 2: tuple.
        train_per2 = .8
        valid_per2 = .1
        data_size2 = 10
        exp_train_size2 = math.ceil(data_size2 * train_per2)
        exp_valid_size2 = math.ceil(data_size2 * valid_per2)
        exp_test_size2 = math.ceil(
            data_size2 * (1 - (train_per2 + valid_per2)))
        shuffle2 = True
        indices2 = tuple([i for i in range(data_size2)])

        tr_idx2, val_idx2, te_idx2 = trainer_helper.generate_splits(
            indices2, train_per=train_per2, valid_per=valid_per2, 
            shuffle=shuffle2)

        self.assertTrue(tr_idx2.size(0) == exp_train_size2)
        self.assertTrue(val_idx2.size(0) == exp_valid_size2)
        self.assertTrue(te_idx2.size(0) == exp_test_size2)

        all_indices2 = tr_idx2.tolist() + val_idx2.tolist() + te_idx2.tolist()
        all_indices2.sort()
        self.assertTrue(tuple(all_indices2) == indices2)

        # Case 3: torch.LongTensor.
        train_per3 = .8
        valid_per3 = .1
        data_size3 = 10
        exp_train_size3 = math.ceil(data_size3 * train_per3)
        exp_valid_size3 = math.ceil(data_size3 * valid_per3)
        exp_test_size3 = math.ceil(
            data_size3 * (1 - (train_per3 + valid_per3)))
        shuffle3 = True
        indices3 = torch.arange(0, data_size3).long()

        tr_idx3, val_idx3, te_idx3 = trainer_helper.generate_splits(
            indices3, train_per=train_per3, valid_per=valid_per3, 
            shuffle=shuffle3)

        self.assertTrue(tr_idx3.size(0) == exp_train_size3)
        self.assertTrue(val_idx3.size(0) == exp_valid_size3)
        self.assertTrue(te_idx3.size(0) == exp_test_size3)

        all_indices3 = tr_idx3.tolist() + val_idx3.tolist() + te_idx3.tolist()
        all_indices3.sort()
        self.assertTrue(tuple(all_indices3) == tuple(indices3.tolist()))

    def test_stratified_kfold_iter_exceptions(self):
        '''
        Test stratified_kfold_iter throws the correct exceptions for bad 
        inputs.
        '''

        # Case 1: Different lengths of indices and labels inputs.
        num_folds1 = 2
        data_size1 = 4
        indices1 = [i for i in range(data_size1)]
        labels1 = [0, 0, 1, 1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_kfold_iter(indices1, labels1, num_folds1)
            self.assertEqual(
                "labels and indices must be the same size.",
                context.exception)
            
        # Case 2: indices is of some other type that is not a list, tuple, 
        # or a torch.LongTensor. For example, generators, while iterables,
        # would possibly break behavior.
        num_folds2 = 2
        data_size2 = 4
        indices2 = set([i for i in range(0, data_size2)])
        labels2 = [0, 0, 1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_kfold_iter(indices2, labels2, num_folds2)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 3: indices contains a negative integer. 
        # This wouldn't technically break the behavior but does not make sense
        # semantically if indices is supposed to correspond to dataset items.
        num_folds3 = 2
        data_size3 = 4
        indices3 = [i for i in range(-1, data_size2 - 1)]
        labels3 = [0, 0, 1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_kfold_iter(indices3, labels3, num_folds3)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 4: labels is of some other type that is not a list, tuple, 
        # or a torch.LongTensor. For example, generators, while iterables,
        # would possibly break behavior.
        num_folds4 = 2
        data_size4 = 4
        indices4 = [i for i in range(0, data_size2)]
        labels4 = set([0, 0, 1, 1])
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_kfold_iter(indices4, labels4, num_folds4)
            self.assertEqual(
                "labels must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 5: labels contains a negative integer. 
        # This wouldn't technically break the behavior but does not make sense
        # semantically if indices is supposed to correspond to dataset items.
        num_folds5 = 2
        data_size5 = 4
        indices5 = [i for i in range(data_size2)]
        labels5 = [0, 0, -1, 1]
        with self.assertRaises(Exception) as context:
            trainer_helper.stratified_kfold_iter(indices5, labels5, num_folds5)
            self.assertEqual(
                "labels must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

    def test_stratified_kfold_iter(self):
        '''
        Test stratified_kfold_iter for cases with and without a validation set.
        '''

        # Case 1: Without validation set.
        num_folds1 = 2
        data_size1 = 4 
        valid_per1 = 0
        indices1 = [i for i in range(data_size1)]
        labels1 = [0, 0, 1, 1]
        
        kfold_iter1 = trainer_helper.stratified_kfold_iter(
            indices1, labels1, num_folds1, valid_per=valid_per1)

        for tr_idx, te_idx in kfold_iter1:
            
            train_label_counts1 = defaultdict(int)
            for idx in tr_idx:
                train_label_counts1[labels1[idx]] += 1
            for label, count in train_label_counts1.items():
                self.assertTrue(count == 1)
            
            test_label_counts1 = defaultdict(int)
            for idx in te_idx:
                test_label_counts1[labels1[idx]] += 1
            for label, count in test_label_counts1.items():
                self.assertTrue(count == 1)

        # Case 2: With validation set.
        num_folds2 = 3
        data_size2 = 12
        indices2 = [i for i in range(data_size2)]
        labels2 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        valid_per2 = .1
        
        kfold_iter2 = trainer_helper.stratified_kfold_iter(
            indices2, labels2, num_folds2, valid_per=valid_per2)

        for tr_idx, val_idx, te_idx in kfold_iter2:
            
            train_label_counts2 = defaultdict(int)
            for idx in tr_idx:
                train_label_counts2[labels2[idx]] += 1
            for label, count in train_label_counts2.items():
                self.assertTrue(count == 3)

            valid_label_counts2 = defaultdict(int)
            for idx in val_idx:
                valid_label_counts2[labels2[idx]] += 1
            for label, count in valid_label_counts2.items():
                self.assertTrue(count == 1)

            test_label_counts2 = defaultdict(int)
            for idx in te_idx:
                test_label_counts2[labels2[idx]] += 1
            for label, count in test_label_counts2.items():
                self.assertTrue(count == 2)


    def test_kfold_iter_shuffle(self):
        '''
        Test that kfold_iter respects no shuffling option.
        Note that if valid_per is greater than 0 it is expected behavior that 
        the training and validation indices will not be in order since 
        the validation set is randomly chosen from the training set at each 
        fold. The test set should still be in order.
        '''
        num_folds1 = 5
        data_size1 = 5
        indices1 = [i for i in range(0, data_size1)]
        
        fold_iter = trainer_helper.kfold_iter(
            indices1, num_folds1, valid_per=0, shuffle=False)

        for i, (tr_idx, te_idx) in enumerate(fold_iter):
            # Next asserts require te_idx to be of size 1, so make sure this
            # holds here.
            self.assertTrue(te_idx.size(0) == 1)
            
            # Check that test_idx is in correct order 
            # (i.e. no shuffling happened).
            self.assertTrue(te_idx[0] == indices1[i])

            # Check that train_idx is in correct_order.
            for j in range(tr_idx.size(0)):
                real_idx = j + 1 if j >= i else j
                self.assertTrue(tr_idx[j] == indices1[real_idx])

    def test_kfold_iter_exceptions(self):
        '''
        Test that kfold_iter throughs the expected exceptions for bad inputs.
        '''

        # Case 1: indices contains a negative integer. 
        # This wouldn't technically break the behavior but does not make sense
        # semantically if indices is supposed to correspond to dataset items.
        num_folds1 = 3
        data_size1 = 5
        indices1 = [i for i in range(-5, data_size1)]
        with self.assertRaises(Exception) as context:
            trainer_helper.kfold_iter(indices1, num_folds1)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)
 
        # Case 2: indices is of some other type that is not a list, tuple, 
        # or a torch.LongTensor. For example, generators, while iterables,
        # would possibly break behavior.
        num_folds2 = 3
        data_size2 = 5
        indices2 = set([i for i in range(0, data_size2)])
        with self.assertRaises(Exception) as context:
            trainer_helper.kfold_iter(indices2, num_folds2)
            self.assertEqual(
                "indices must be a list, tuple, or torch.LongTensor of " \
                "nonnegative integers.",
                context.exception)

        # Case 3: Number of folds is too big for the dataset.
        num_folds3 = 7
        data_size3 = 5
        indices3 = set([i for i in range(0, data_size3)])
        with self.assertRaises(Exception) as context:
            trainer_helper.kfold_iter(indices3, num_folds3)
            self.assertEqual(
                'Not enough data points for {} folds.'.format(num_folds5),
                context.exception)

        # Case 4: valid_per is not a float in (0, 1].
        num_folds4 = 3
        data_size4 = 5
        valid_per4 = -.1
        indices4 = set([i for i in range(0, data_size4)])
        with self.assertRaises(Exception) as context:
            trainer_helper.kfold_iter(
                indices4, num_folds4, valid_per=valid_per4)
            self.assertEqual(
                "valid_per must be float in range (0, 1]",
                context.exception)

    def test_kfold_iter_with_valid(self):
        '''
        Test that kfold_iter returns the correctly sized training, validation,
        and testing paritions on either list, tuple, or torch.LongTensor.
        '''

        # Case 1: list.
        num_folds1 = 3
        data_size1 = 5
        valid_per1 = .1
        exp_fold_size1 = math.ceil(data_size1 / num_folds1)
        exp_last_fold_size1 = data_size1 - (num_folds1 - 1) * exp_fold_size1
        exp_val_size1 = math.ceil(exp_fold_size1 * valid_per1)
        exp_train_size1 = data_size1 - exp_fold_size1 - exp_val_size1
        exp_last_train_size1 = data_size1 - exp_last_fold_size1 - exp_val_size1

        indices1 = [i for i in range(data_size1)]
        fold_iter1 = trainer_helper.kfold_iter(
            indices1, num_folds1, valid_per=valid_per1)
        
        found_test_indices1 = []

        for kfold1, (tr_idx, val_idx, te_idx) in enumerate(fold_iter1, 1):
            for idx in te_idx:
                self.assertFalse(idx in tr_idx)
                self.assertFalse(idx in val_idx)
                found_test_indices1.append(idx)
            for idx in val_idx:
                self.assertFalse(idx in tr_idx)
            all_idx = torch.cat([tr_idx, val_idx, te_idx]).tolist()
            all_idx.sort()
            self.assertTrue(tuple(all_idx) == tuple(indices1))

            if kfold1 < num_folds1:
                self.assertTrue(te_idx.size(0) == exp_fold_size1)
                self.assertTrue(tr_idx.size(0) == exp_train_size1)
            else:
                self.assertTrue(te_idx.size(0) == exp_last_fold_size1)
                self.assertTrue(tr_idx.size(0) == exp_last_train_size1)
            self.assertTrue(val_idx.size(0) == exp_val_size1)

        self.assertTrue(kfold1 == num_folds1)
        found_test_indices1.sort()
        self.assertTrue(tuple(found_test_indices1) == tuple(indices1))

        # Case 2: tuple.
        num_folds2 = 3
        data_size2 = 5
        valid_per2 = .1
        exp_fold_size2 = math.ceil(data_size2 / num_folds2)
        exp_last_fold_size2 = data_size2 - (num_folds2 - 1) * exp_fold_size2
        exp_val_size2 = math.ceil(exp_fold_size2 * valid_per2)
        exp_train_size2 = data_size2 - exp_fold_size2 - exp_val_size2
        exp_last_train_size2 = data_size2 - exp_last_fold_size2 - exp_val_size2
        indices2 = tuple([i for i in range(data_size2)])
        fold_iter2 = trainer_helper.kfold_iter(
            indices2, num_folds2, valid_per=valid_per2)
        
        found_test_indices2 = []

        for kfold2, (tr_idx, val_idx, te_idx) in enumerate(fold_iter2, 1):
            for idx in te_idx:
                self.assertFalse(idx in tr_idx)
                self.assertFalse(idx in val_idx)
                found_test_indices2.append(idx)
            for idx in val_idx:
                self.assertFalse(idx in tr_idx)
            all_idx = torch.cat([tr_idx, val_idx, te_idx]).tolist()
            all_idx.sort()
            self.assertTrue(tuple(all_idx) == indices2)

            if kfold2 < num_folds2:
                self.assertTrue(te_idx.size(0) == exp_fold_size2)
                self.assertTrue(tr_idx.size(0) == exp_train_size2)
            else:
                self.assertTrue(te_idx.size(0) == exp_last_fold_size2)
                self.assertTrue(tr_idx.size(0) == exp_last_train_size2)
            self.assertTrue(val_idx.size(0) == exp_val_size2)

        self.assertTrue(kfold2 == num_folds2)
        found_test_indices2.sort()
        self.assertTrue(tuple(found_test_indices2) == indices2)
 
        # Case 3: torch.LongTensor.
        num_folds3 = 3
        data_size3 = 5
        valid_per3 = .1
        exp_fold_size3 = math.ceil(data_size3 / num_folds3)
        exp_last_fold_size3 = data_size3 - (num_folds3 - 1) * exp_fold_size3
        exp_val_size3 = math.ceil(exp_fold_size3 * valid_per3)
        exp_train_size3 = data_size3 - exp_fold_size3 - exp_val_size3
        exp_last_train_size3 = data_size3 - exp_last_fold_size3 - exp_val_size3

        indices3 = torch.arange(0, data_size3).long()
        fold_iter3 = trainer_helper.kfold_iter(
            indices3, num_folds3, valid_per=valid_per3)
        
        found_test_indices3 = []

        for kfold3, (tr_idx, val_idx, te_idx) in enumerate(fold_iter3, 1):
            for idx in te_idx:
                self.assertFalse(idx in tr_idx)
                self.assertFalse(idx in val_idx)
                found_test_indices3.append(idx)
            for idx in val_idx:
                self.assertFalse(idx in tr_idx)
            all_idx = torch.cat([tr_idx, val_idx, te_idx]).tolist()
            all_idx.sort()
            self.assertTrue(tuple(all_idx) == tuple(indices3.tolist()))

            if kfold3 < num_folds3:
                self.assertTrue(te_idx.size(0) == exp_fold_size3)
                self.assertTrue(tr_idx.size(0) == exp_train_size3)
            else:
                self.assertTrue(te_idx.size(0) == exp_last_fold_size3)
                self.assertTrue(tr_idx.size(0) == exp_last_train_size3)
            self.assertTrue(val_idx.size(0) == exp_val_size3)

        self.assertTrue(kfold3 == num_folds3)
        found_test_indices3.sort()
        self.assertTrue(tuple(found_test_indices3) == tuple(indices3.tolist()))
 
    def test_kfold_iter_no_valid(self):
        '''
        Test that kfold_iter with valid_per=0 returns the correctly sized 
        training and testing paritions on either list, tuple, or 
        torch.LongTensor.
        '''

        # Case 1: list.
        num_folds1 = 3
        data_size1 = 5
        valid_per1 = 0
        exp_fold_size1 = math.ceil(data_size1 / num_folds1)
        exp_last_fold_size1 = data_size1 - (num_folds1 - 1) * exp_fold_size1
        exp_train_size1 = data_size1 - exp_fold_size1 
        exp_last_train_size1 = data_size1 - exp_last_fold_size1 

        indices1 = [i for i in range(data_size1)]
        fold_iter1 = trainer_helper.kfold_iter(
            indices1, num_folds1, valid_per=valid_per1)
        
        found_test_indices1 = []

        for kfold1, (tr_idx, te_idx) in enumerate(fold_iter1, 1):
            for idx in te_idx:
                self.assertFalse(idx in tr_idx)
                found_test_indices1.append(idx)
            all_idx = torch.cat([tr_idx, te_idx]).tolist()
            all_idx.sort()
            self.assertTrue(tuple(all_idx) == tuple(indices1))

            if kfold1 < num_folds1:
                self.assertTrue(te_idx.size(0) == exp_fold_size1)
                self.assertTrue(tr_idx.size(0) == exp_train_size1)
            else:
                self.assertTrue(te_idx.size(0) == exp_last_fold_size1)
                self.assertTrue(tr_idx.size(0) == exp_last_train_size1)

        self.assertTrue(kfold1 == num_folds1)
        found_test_indices1.sort()
        self.assertTrue(tuple(found_test_indices1) == tuple(indices1))
        
        # Case 2: tuple.
        num_folds2 = 3
        data_size2 = 5
        valid_per2 = 0
        exp_fold_size2 = math.ceil(data_size2 / num_folds2)
        exp_last_fold_size2 = data_size2 - (num_folds2 - 1) * exp_fold_size2
        exp_train_size2 = data_size2 - exp_fold_size2
        exp_last_train_size2 = data_size2 - exp_last_fold_size2

        indices2 = tuple([i for i in range(data_size2)])
        fold_iter2 = trainer_helper.kfold_iter(
            indices2, num_folds2, valid_per=valid_per2)
        
        found_test_indices2 = []

        for kfold2, (tr_idx, te_idx) in enumerate(fold_iter2, 1):
            for idx in te_idx:
                self.assertFalse(idx in tr_idx)
                found_test_indices2.append(idx)
            all_idx = torch.cat([tr_idx, te_idx]).tolist()
            all_idx.sort()
            self.assertTrue(tuple(all_idx) == indices2)

            if kfold2 < num_folds2:
                self.assertTrue(te_idx.size(0) == exp_fold_size2)
                self.assertTrue(tr_idx.size(0) == exp_train_size2)
            else:
                self.assertTrue(te_idx.size(0) == exp_last_fold_size2)
                self.assertTrue(tr_idx.size(0) == exp_last_train_size2)

        self.assertTrue(kfold2 == num_folds2)
        found_test_indices2.sort()
        self.assertTrue(tuple(found_test_indices2) == indices2)
 
        # Case 3: torch.LongTensor.
        num_folds3 = 3
        data_size3 = 5
        valid_per3 = 0
        exp_fold_size3 = math.ceil(data_size3 / num_folds3)
        exp_last_fold_size3 = data_size3 - (num_folds3 - 1) * exp_fold_size3
        exp_train_size3 = data_size3 - exp_fold_size3
        exp_last_train_size3 = data_size3 - exp_last_fold_size3

        indices3 = torch.arange(0, data_size3).long()
        fold_iter3 = trainer_helper.kfold_iter(
            indices3, num_folds3, valid_per=valid_per3)
        
        found_test_indices3 = []

        for kfold3, (tr_idx, te_idx) in enumerate(fold_iter3, 1):
            for idx in te_idx:
                self.assertFalse(idx in tr_idx)
                found_test_indices3.append(idx)
            all_idx = torch.cat([tr_idx, te_idx]).tolist()
            all_idx.sort()
            self.assertTrue(tuple(all_idx) == tuple(indices3.tolist()))

            if kfold3 < num_folds3:
                self.assertTrue(te_idx.size(0) == exp_fold_size3)
                self.assertTrue(tr_idx.size(0) == exp_train_size3)
            else:
                self.assertTrue(te_idx.size(0) == exp_last_fold_size3)
                self.assertTrue(tr_idx.size(0) == exp_last_train_size3)

        self.assertTrue(kfold3 == num_folds3)
        found_test_indices3.sort()
        self.assertTrue(tuple(found_test_indices3) == tuple(indices3.tolist()))
        
if __name__ == '__main__':
    unittest.main()
