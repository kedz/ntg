import unittest
from nt.dataio import Dataset
import torch

class TestDataset(unittest.TestCase):

    def test_no_layout_constr(self):
        feature1 = torch.LongTensor([[1, 2], 
                                     [3, 4], 
                                     [5, 6]])

        feature2 = torch.FloatTensor([[1.5, 2.5], 
                                      [3.5, 4.5], 
                                      [5.5, 6.5]])

        labels = torch.LongTensor([1, 2, 3])


        data = [(feature1, "feature1"), (feature2, "feature2"), 
                (labels, "targets")]

        dataset = Dataset(*data, batch_size=1, shuffle=False, gpu=-1)

        self.assertTrue(hasattr(dataset, "feature1"))
        self.assertTrue(hasattr(dataset, "feature2"))
        self.assertTrue(hasattr(dataset, "targets"))

        self.assertTrue(dataset.feature1.equal(feature1))
        self.assertTrue(dataset.feature2.equal(feature2))
        self.assertTrue(dataset.targets.equal(labels))


    def test_attribute_getters(self):

        feature1 = torch.LongTensor([[1, 2], 
                                     [3, 4], 
                                     [5, 6]])

        feature2 = torch.FloatTensor([[1.5, 2.5], 
                                      [3.5, 4.5], 
                                      [5.5, 6.5]])

        labels = torch.LongTensor([1, 2, 3])

        layout = [["inputs", [["feature1", "feature1"], 
                              ["feature2", "feature2"]]],
                  ["targets", "targets"]]

        data = [(feature1, "feature1"), (feature2, "feature2"), 
                (labels, "targets")]

        dataset = Dataset(
            *data, layout=layout, batch_size=1, shuffle=False, gpu=-1)

        self.assertTrue(hasattr(dataset, "inputs"))
        self.assertTrue(hasattr(dataset.inputs, "feature1"))
        self.assertTrue(hasattr(dataset.inputs, "feature2"))
        self.assertTrue(hasattr(dataset, "targets"))

        self.assertTrue(dataset.inputs.feature1.equal(feature1))
        self.assertTrue(dataset.inputs.feature2.equal(feature2))
        self.assertTrue(dataset.targets.equal(labels))

        
    def test_index_select(self):

        feature1 = torch.LongTensor([[1 ,2], 
                                     [3, 4], 
                                     [5, 6]])

        feature2 = torch.FloatTensor([[1.5, 2.5], 
                                      [3.5, 4.5], 
                                      [5.5, 6.5]])

        labels = torch.LongTensor([1, 2, 3])

        layout = [["inputs", [["feature1", "feature1"], 
                              ["feature2", "feature2"]]],
                  ["targets", "targets"]]

        data = [(feature1, "feature1"), (feature2, "feature2"), 
                (labels, "targets")]

        dataset = Dataset(
            *data, layout=layout, batch_size=1, shuffle=False, gpu=-1)
        index = torch.LongTensor([0, 2])
        subset = dataset.index_select(index)
        
        self.assertTrue(
            subset.inputs.feature1.equal(feature1.index_select(0, index)))


    def test_iter_batch_no_shuffle_no_length(self):

        feature1 = torch.LongTensor([[ 1,  2], 
                                     [ 3,  4], 
                                     [ 5,  6],
                                     [ 7,  8],
                                     [ 9, 10],
                                     [11, 12],
                                     [13, 14],
                                     [15, 16],
                                     [17, 18],
                                     [19, 20],
                                     [21, 22]])

        labels = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        layout = [["inputs", "inputs"], ["targets", "targets"]]

        data = [(feature1, "inputs"), (labels, "targets")]

        bs = 3

        dataset = Dataset(
            *data, layout=layout, batch_size=bs, shuffle=False, gpu=-1)
        index = torch.LongTensor([i for i in range(11)])

        for batch_num, batch in enumerate(dataset.iter_batch()):
            
            if batch_num < labels.size(0) // bs:
                self.assertTrue(batch.inputs.size(0) == bs)
            else:
                self.assertTrue(batch.inputs.size(0) == labels.size(0) % bs)
            self.assertTrue(
                feature1[batch_num * bs: (batch_num + 1) * bs].equal(
                    batch.inputs.data))
            self.assertTrue(
                labels[batch_num * bs: (batch_num + 1) * bs].equal(
                    batch.targets.data))
             
    def test_iter_batch_shuffle_no_length(self):

        feature1 = torch.LongTensor([[ 1,  2], 
                                     [ 3,  4], 
                                     [ 5,  6],
                                     [ 7,  8],
                                     [ 9, 10],
                                     [11, 12],
                                     [13, 14],
                                     [15, 16],
                                     [17, 18],
                                     [19, 20],
                                     [21, 22]])

        labels = torch.LongTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

        layout = [["inputs", "inputs"], ["targets", "targets"]]

        data = [(feature1, "inputs"), (labels, "targets")]

        bs = 3

        dataset = Dataset(*data, layout=layout, batch_size=bs, shuffle=True, gpu=-1)
        index = torch.LongTensor([i for i in range(11)])

        found_labels = []

        for batch_num, batch in enumerate(dataset.iter_batch()):
            
            if batch_num < labels.size(0) // bs:
                self.assertTrue(batch.inputs.size(0) == bs)
            else:
                self.assertTrue(batch.inputs.size(0) == labels.size(0) % bs)
            self.assertFalse(
                feature1[batch_num * bs: (batch_num + 1) * bs].equal(
                    batch.inputs.data))
            self.assertFalse(
                labels[batch_num * bs: (batch_num + 1) * bs].equal(
                    batch.targets.data))
            for label in batch.targets.data:
                found_labels.append(label)
        
        found_labels.sort()
        self.assertTrue(torch.LongTensor(found_labels).equal(labels))


if __name__ == '__main__':
    unittest.main()
