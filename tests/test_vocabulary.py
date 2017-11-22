import unittest
import nt

class TestVocabulary(unittest.TestCase):

    def test_top_k(self):
        vocab = nt.dataio.Vocabulary(top_k=10)
        self.assertTrue(vocab.top_k == 10)
        vocab.top_k = 15
        self.assertTrue(vocab.top_k == 15)
        with self.assertRaises(Exception) as context:
            vocab.top_k = 0
        self.assertEqual(
            "top_k must be a positive int.", str(context.exception))

        with self.assertRaises(Exception) as context:
            vocab.top_k = 4.0
        self.assertEqual(
            "top_k must be a positive int.", str(context.exception))

    def test_at_least(self):
        vocab = nt.dataio.Vocabulary(at_least=3)
        self.assertTrue(vocab.at_least == 3)
        vocab.at_least = 15
        self.assertTrue(vocab.at_least == 15)
        with self.assertRaises(Exception) as context:
            vocab.at_least = 0
        self.assertEqual(
            "at_least must be a positive int.", str(context.exception))

        with self.assertRaises(Exception) as context:
            vocab.at_least = 4.0
        self.assertEqual(
            "at_least must be a positive int.", str(context.exception))

        vocab2 = nt.dataio.Vocabulary(at_least=2, unknown_token="__UNK__",
            special_tokens=("A", "B"), zero_indexing=False)
        vocab2["a"]
        vocab2["a"]
        vocab2["b"]
        vocab2.freeze()

        self.assertTrue(vocab2.unknown_index == 1)
        self.assertTrue(len(vocab2) == 4)
        self.assertTrue(vocab2["__UNK__"] == 1)
        self.assertTrue(vocab2["A"] == 2)
        self.assertTrue(vocab2["B"] == 3)
        self.assertTrue(vocab2["a"] == 4)
        self.assertTrue(vocab2["b"] == vocab2.unknown_index)
        self.assertTrue(vocab2[1] == "__UNK__")
        self.assertTrue(vocab2[2] == "A")
        self.assertTrue(vocab2[3] == "B")
        self.assertTrue(vocab2[4] == "a")
        self.assertTrue(vocab2[vocab2.unknown_index] == "__UNK__")

    def test_unknown_token(self):
        vocab = nt.dataio.Vocabulary(unknown_token="__UNK__")
        self.assertTrue(vocab.unknown_token == "__UNK__")
        self.assertTrue(vocab.unknown_index == 0)

        vocab["a"]
        vocab.freeze()
        self.assertTrue(vocab["a"] != vocab.unknown_index)
        self.assertTrue(vocab["b"] == vocab.unknown_index)

        vocab2 = nt.dataio.Vocabulary(
            unknown_token="__UNK__", zero_indexing=False)
        self.assertTrue(vocab2.unknown_token == "__UNK__")
        self.assertTrue(vocab2.unknown_index == 1)
        vocab2["c"]
        vocab2.freeze()
        self.assertTrue(vocab2["c"] != vocab2.unknown_index)
        self.assertTrue(vocab2["d"] == vocab2.unknown_index)

        vocab3 = nt.dataio.Vocabulary()

        vocab3["e"]
        vocab3.freeze()
        self.assertTrue(vocab3["f"] is None)

    def test_special_tokens(self):
        vocab1 = nt.dataio.Vocabulary(special_tokens="__START__")
        self.assertTrue(vocab1.special_tokens == ("__START__",))
        vocab2 = nt.dataio.Vocabulary(
            top_k=2, special_tokens=["__START__", "__STOP__"])
        self.assertTrue(vocab2.special_tokens == ("__START__", "__STOP__"))

        self.assertTrue(vocab2["__START__"] == 0)
        self.assertTrue(vocab2["__STOP__"] == 1)
        for i in range(4): 
            vocab2["a"]
        for i in range(3): 
            vocab2["b"]
        for i in range(2): 
            vocab2["c"]
        vocab2.freeze()
        self.assertTrue(vocab2["__START__"] == 0)
        self.assertTrue(vocab2["__STOP__"] == 1)
        self.assertTrue(vocab2["a"] == 2)
        self.assertTrue(vocab2["b"] == 3)
        self.assertTrue(vocab2["c"] == None)

    def test_iter(self):
        vocab1 = nt.dataio.Vocabulary(
            special_tokens="__START__", zero_indexing=False)
        vocab1["a"]
        vocab1["b"]
        vocab1["b"]
        vocab1.freeze()
        self.assertTrue(
            tuple([token for token in vocab1]) == ("__START__", "b", "a"))

        vocab2 = nt.dataio.Vocabulary(
            special_tokens="__START__", zero_indexing=True)
        vocab2["a"]
        vocab2["b"]
        vocab2["b"]
        vocab2.freeze()
        self.assertTrue(
            tuple([token for token in vocab2]) == ("__START__", "b", "a"))

    def test_to_string(self):    
        vocab1 = nt.dataio.Vocabulary(
            special_tokens="__START__", zero_indexing=False)
        vocab1["a"]
        vocab1["a"]
        vocab1["b"]
        self.assertTrue(str(vocab1) == "Vocabulary(size=3)")
                
if __name__ == '__main__':
    unittest.main()
