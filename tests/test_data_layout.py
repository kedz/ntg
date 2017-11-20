import unittest
from dataset import DataLayout
import torch

class TestDataLayout(unittest.TestCase):

    def test_attributes(self):
        layout = [["inputs", [["encoder_input", "A"], ["decoder_input", "B"]]],
                  ["outputs", [["decoder_output", "B"]]]]
        
        label2data = {"A": torch.FloatTensor([3.14]),
                      "B": torch.FloatTensor([666])}
        dl = DataLayout(layout, label2data)

        self.assertTrue(hasattr(dl, "inputs"))
        self.assertTrue(hasattr(dl, "outputs"))
        self.assertTrue(hasattr(dl.inputs, "encoder_input"))
        self.assertTrue(hasattr(dl.inputs, "decoder_input"))
        self.assertTrue(hasattr(dl.outputs, "decoder_output"))
    
    def test_storage(self):
        layout = [["inputs", [["encoder_input", "A"], ["decoder_input", "B"]]],
                  ["outputs", [["decoder_output", "B"]]]]

        label2data = {"A": torch.FloatTensor([3.14]),
                      "B": torch.FloatTensor([666])}
        dl = DataLayout(layout, label2data)
        
        label2data["A"].fill_(-99)
        label2data["B"].fill_(42)

        self.assertTrue(dl.inputs.encoder_input.equal(label2data["A"]))
        self.assertTrue(dl.inputs.decoder_input.equal(label2data["B"]))
        self.assertTrue(dl.outputs.decoder_output.equal(label2data["B"]))

    def test_iterator(self):
        
        layout = [["inputs", [["encoder_input", "A"], 
                              ["decoder_input", "B"],
                              ["decoder_output", "A"]]]]
        
        label2data = {"A": torch.FloatTensor([3.14]),
                      "B": torch.FloatTensor([666])}
                      
        dl = DataLayout(layout, label2data)
        for x, y in zip(dl.inputs, 
                        [label2data["A"], label2data["B"], label2data["A"]]):
            self.assertTrue(x.equal(y))

        layout = [["encoder_input", "A"], 
                  ["decoder_input", "B"],
                  ["decoder_output", "A"]]
        
        dl = DataLayout(layout, label2data)
        for x, y in zip(dl, 
                        [label2data["A"], label2data["B"], label2data["A"]]):
            self.assertTrue(x.equal(y))

    def test_index(self):       
        label2data = {"A": torch.FloatTensor([3.14]),
                      "B": torch.FloatTensor([666])}
                      
        layout = [["encoder_input", "A"], 
                  ["decoder_input", "B"],
                  ["decoder_output", "A"]]
        
        dl = DataLayout(layout, label2data)

        self.assertTrue(dl[0].equal(label2data["A"]))
        self.assertTrue(dl[1].equal(label2data["B"]))
        self.assertTrue(dl[2].equal(label2data["A"]))

    def test_tensor_indexing(self):
        layout = [["inputs", [["encoder_input", "A"], ["decoder_input", "B"]]],
                  ["outputs", [["decoder_output", "B"]]]]
        
        A = torch.FloatTensor([[1,2,3], [4,5,6], [7, 8, 9]])
        B = torch.FloatTensor([6, 15, 24])
        label2data = {"A": A, "B": B}

        dl = DataLayout(layout, label2data)

        index = torch.LongTensor([0, 2])
        dl2 = dl.index_select(index)

        self.assertTrue(
            dl2.inputs.encoder_input.equal(A.index_select(0, index)))
        self.assertTrue(
            dl2.inputs.decoder_input.equal(B.index_select(0, index)))
        self.assertTrue(
            dl2.outputs.decoder_output.equal(B.index_select(0, index)))


if __name__ == '__main__':
    unittest.main()
