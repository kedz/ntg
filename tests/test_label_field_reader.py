import unittest
import nt
import torch

class TestLabelFieldReader(unittest.TestCase):

    def test_preset_vocab(self):

        data1 = {"label": "positive"}
        data2 = {"label": "negative"}
        data3 = {"label": "positive"}

        preset1 = ["negative", "positive"]
        label_reader1 = nt.dataio.field_reader.Label(
            "label", vocabulary=preset1)

        self.assertTrue(label_reader1.labels == tuple(preset1))
        label_reader1.read(data1)
        label_reader1.read(data2)
        label_reader1.read(data3)

        label_reader1.fit_parameters()
        self.assertTrue(label_reader1.labels == tuple(preset1))

        data4 = {"label": "neutral"}
        with self.assertRaises(Exception) as context:
            label_reader1.read(data4)
        self.assertEqual(
            "Found unknown label string: neutral",
            str(context.exception))

    def test_no_zero_indexing(self):
        data1 = {"label": "positive"}
        data2 = {"label": "negative"}
        data3 = {"label": "negative"}

        label_reader1 = nt.dataio.field_reader.Label(
            "label", zero_indexing=False)
        label_reader1.read(data1)
        label_reader1.read(data2)
        label_reader1.read(data3)
        result1, = label_reader1.finish_read()
        self.assertTrue(result1.equal(torch.LongTensor([1, 2, 2])))
        self.assertTrue(
            tuple(["positive", "negative"]) == label_reader1.labels)
        label_reader1.fit_parameters()
        self.assertTrue(
            tuple(["negative", "positive"]) == label_reader1.labels)

        label_reader1.read(data1)
        label_reader1.read(data2)
        label_reader1.read(data3)
        result2, = label_reader1.finish_read()
        self.assertTrue(result2.equal(torch.LongTensor([2, 1, 1])))

    def test_read_ints(self):
        data1 = {"label": 1}
        data2 = {"label": 2}
        data3 = {"label": 1}

        label_reader1 = nt.dataio.field_reader.Label("label")
        label_reader1.read(data1)
        label_reader1.read(data2)
        label_reader1.read(data3)
        result1, = label_reader1.finish_read()
        self.assertTrue(result1.equal(torch.LongTensor([0, 1, 0])))

        self.assertTrue(tuple(["1", "2"]) == label_reader1.labels)

    def test_read(self):

        data1 = [[42.3, 64.], "positive"]
        data2 = [[42.3, 64.], "negative"]
        data3 = [[42.3, 64.], "negative"]
        
        label_reader = nt.dataio.field_reader.Label(1)
        label_reader.read(data1)
        label_reader.read(data2)
        label_reader.read(data3)
        
        result1, = label_reader.finish_read()
        self.assertTrue(result1.equal(torch.LongTensor([0, 1, 1])))
        
        self.assertTrue(label_reader.labels == tuple(["positive", "negative"])) 
        label_reader.fit_parameters()

        label_reader.read(data1)
        label_reader.read(data2)
        label_reader.read(data3)
        result2, = label_reader.finish_read(reset=False)
        self.assertTrue(result2.equal(torch.LongTensor([1, 0, 0])))

        label_reader.vector_type = float
        result3, = label_reader.finish_read(reset=False)
        self.assertTrue(result3.equal(torch.FloatTensor([1, 0, 0])))

        label_reader.vector_type = bytes
        result4, = label_reader.finish_read()
        self.assertTrue(result4.equal(torch.ByteTensor([1, 0, 0])))

        self.assertTrue(label_reader.labels == tuple(["negative", "positive"])) 

                
if __name__ == '__main__':
    unittest.main()
