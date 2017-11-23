import unittest
import nt
import tempfile
import os
import torch


class TestXSVReader(unittest.TestCase):

    def setUp(self):
        data = "\n".join(["h1,h2,label",
                          "1.5,2.5,pos",
                          "2.3,3.14,neg",
                          "3.33,7.5,neg"])

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(data)
            self.csv_header_path = fp.name

        data = "\n".join(["1.5,2.5,pos",
                          "2.3,3.14,neg",
                          "3.33,7.5,neg"])

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(data)
            self.csv_no_header_path = fp.name

    def tearDown(self):
        os.remove(self.csv_header_path)
        os.remove(self.csv_no_header_path)

    def test_read_header(self):

        f1 = nt.dataio.field_reader.DenseVector("h1")
        f2 = nt.dataio.field_reader.DenseVector("h2")
        f3 = nt.dataio.field_reader.Label("label")
        fields = [f1, f2, f3]
        file_reader = nt.dataio.file_reader.XSVReader(
            fields, ",", skip_header=True)
        (data1,), (data2,), (data3,) = file_reader.read(self.csv_header_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([0, 1, 1])))

        file_reader.fit_parameters()
        (data1,), (data2,), (data3,) = file_reader.read(self.csv_header_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([1, 0, 0])))

        f4 = nt.dataio.field_reader.DenseVector(0)
        f5 = nt.dataio.field_reader.DenseVector(1)
        f6 = nt.dataio.field_reader.Label(2)
        fields2 = [f4, f5, f6]
        file_reader2 = nt.dataio.file_reader.XSVReader(
            fields2, ",", skip_header=True)
        file_reader2.fit_parameters(self.csv_header_path)
        self.assertTrue(f6.labels == tuple(["neg", "pos"]))

        (data4,), (data5,), (data6,) = file_reader2.read(self.csv_header_path)

        self.assertTrue(data4.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data5.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data6.equal(torch.LongTensor([1, 0, 0])))


    def test_read_no_header(self):
        f1 = nt.dataio.field_reader.DenseVector(0)
        f2 = nt.dataio.field_reader.DenseVector(1)
        f3 = nt.dataio.field_reader.Label(2)
        fields = [f1, f2, f3]
        file_reader = nt.dataio.file_reader.XSVReader(
            fields, ",", skip_header=False)
        (data1,), (data2,), (data3,) = file_reader.read(
            self.csv_no_header_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([0, 1, 1])))

        file_reader.fit_parameters()
        (data1,), (data2,), (data3,) = file_reader.read(
            self.csv_no_header_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([1, 0, 0])))

if __name__ == '__main__':
    unittest.main()
