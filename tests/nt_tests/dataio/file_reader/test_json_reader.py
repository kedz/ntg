import unittest
import nt
import tempfile
import os
import torch
import json


class TestJSONReader(unittest.TestCase):

    def setUp(self):
        data1 = {"h1": 1.5, "h2": 2.5, "label": "pos"}
        data2 = {"h1": 2.3, "h2": 3.14, "label": "neg"}
        data3 = {"h1": 3.33, "h2": 7.5, "label": "neg"}


        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(json.dumps(data1))
            fp.write("\n")
            fp.write(json.dumps(data2))
            fp.write("\n")
            fp.write(json.dumps(data3))
            fp.write("\n")
            self.line_sep_json_path = fp.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(json.dumps([data1, data2, data3]))
            fp.write("\n")
            self.json_path = fp.name


        as_list = [[data1["h1"], data1["h2"], data1["label"]],
                   [data2["h1"], data2["h2"], data2["label"]],
                   [data3["h1"], data3["h2"], data3["label"]]]

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(json.dumps(as_list))
            fp.write("\n")
            self.json_list_path = fp.name

    def tearDown(self):
        os.remove(self.line_sep_json_path)
        os.remove(self.json_path)
        os.remove(self.json_list_path)

    def test_read_line_sep_json(self):

        f1 = nt.dataio.field_reader.DenseVector("h1")
        f2 = nt.dataio.field_reader.DenseVector("h2")
        f3 = nt.dataio.field_reader.Label("label")
        fields = [f1, f2, f3]
        file_reader = nt.dataio.file_reader.JSONReader(fields)
        (data1,), (data2,), (data3,) = file_reader.read(
            self.line_sep_json_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([0, 1, 1])))

        file_reader.fit_parameters()
        (data1,), (data2,), (data3,) = file_reader.read(
            self.line_sep_json_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([1, 0, 0])))


    def test_read_json_doc(self):
        f1 = nt.dataio.field_reader.DenseVector("h1")
        f2 = nt.dataio.field_reader.DenseVector("h2")
        f3 = nt.dataio.field_reader.Label("label")
        fields = [f1, f2, f3]
        file_reader = nt.dataio.file_reader.JSONReader(
            fields, line_separated=False)
        (data1,), (data2,), (data3,) = file_reader.read(
            self.json_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([0, 1, 1])))

        file_reader.fit_parameters()
        (data1,), (data2,), (data3,) = file_reader.read(
            self.json_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([1, 0, 0])))

    def test_read_json_list(self):
        f1 = nt.dataio.field_reader.DenseVector(0)
        f2 = nt.dataio.field_reader.DenseVector(1)
        f3 = nt.dataio.field_reader.Label(2)
        fields = [f1, f2, f3]
        file_reader = nt.dataio.file_reader.JSONReader(
            fields, line_separated=False)
        (data1,), (data2,), (data3,) = file_reader.read(
            self.json_list_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([0, 1, 1])))

        file_reader.fit_parameters()
        (data1,), (data2,), (data3,) = file_reader.read(
            self.json_list_path)

        self.assertTrue(data1.equal(torch.FloatTensor([[1.5], [2.3], [3.33]])))
        self.assertTrue(data2.equal(torch.FloatTensor([[2.5], [3.14], [7.5]])))
        self.assertTrue(data3.equal(torch.LongTensor([1, 0, 0])))

if __name__ == '__main__':
    unittest.main()
