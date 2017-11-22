import unittest
import nt
import torch

class TestDenseVector(unittest.TestCase):

    def test_read_dict(self):
        data1 = {"features": [1.0, 2.0, 3.0], "label": 1}
        data2 = {"features": [4.0, 5.0, 6.0], "label": 0}
        vector_field = nt.dataio.field_reader.DenseVector("features")
        
        self.assertTrue(vector_field.expected_size == None)
        vector_field.read(data1)
        self.assertTrue(vector_field.expected_size == 3)
        vector_field.read(data2)

        result1, = vector_field.finish_read(reset=False)
        expected1 = torch.FloatTensor([data1["features"], data2["features"]])

        self.assertTrue(torch.equal(result1, expected1))

        vector_field.vector_type = int
        result2, = vector_field.finish_read(reset=False)
        self.assertTrue(torch.equal(result2, expected1.long()))

        vector_field.vector_type = bytes
        result3, = vector_field.finish_read(reset=False)
        self.assertTrue(torch.equal(result3, expected1.byte()))

        vector_field2 = nt.dataio.field_reader.DenseVector(
            "features", expected_size=4)
        with self.assertRaises(Exception) as context:
            vector_field2.read(data1)
        self.assertEqual(
            "Found vector of size 3 but expecting 4",
            str(context.exception))

        vector_field3 = nt.dataio.field_reader.DenseVector(
            "features", expected_size=4, sep=",")
        data3 = {"features": "1.0, 2.0, 3.0, 4.0", "label": 1}
        data4 = {"features": "4.0, 5.0, 6.0, 7.0", "label": 0}
        expected4 = torch.FloatTensor(
            [[float(x) for x in data3["features"].split(",")],
             [float(x) for x in data4["features"].split(",")]])
        
        vector_field3.read(data3)
        vector_field3.read(data4)
        result4, = vector_field3.finish_read()

        self.assertTrue(torch.equal(result4, expected4))
         
    def test_read_list(self):
        data1 = [[1.0, 2.0, 3.0], 1]
        data2 = [[4.0, 5.0, 6.0], 0]
        vector_field = nt.dataio.field_reader.DenseVector(0)
        
        self.assertTrue(vector_field.expected_size == None)
        vector_field.read(data1)
        self.assertTrue(vector_field.expected_size == 3)
        vector_field.read(data2)

        result1, = vector_field.finish_read(reset=False)
        expected1 = torch.FloatTensor([data1[0], data2[0]])

        self.assertTrue(torch.equal(result1, expected1))

        vector_field.vector_type = int
        result2, = vector_field.finish_read(reset=False)
        self.assertTrue(torch.equal(result2, expected1.long()))

        vector_field.vector_type = bytes
        result3, = vector_field.finish_read(reset=False)
        self.assertTrue(torch.equal(result3, expected1.byte()))

        vector_field2 = nt.dataio.field_reader.DenseVector(
            0, expected_size=4)
        with self.assertRaises(Exception) as context:
            vector_field2.read(data1)
        self.assertEqual(
            "Found vector of size 3 but expecting 4",
            str(context.exception))

        vector_field3 = nt.dataio.field_reader.DenseVector(
            0, expected_size=4, sep=",")
        data3 = ["1.0, 2.0, 3.0, 4.0", 1]
        data4 = ["4.0, 5.0, 6.0, 7.0", 0]
        expected4 = torch.FloatTensor(
            [[float(x) for x in data3[0].split(",")],
             [float(x) for x in data4[0].split(",")]])
        
        vector_field3.read(data3)
        vector_field3.read(data4)
        result4, = vector_field3.finish_read()

        self.assertTrue(torch.equal(result4, expected4))

if __name__ == '__main__':
    unittest.main()
