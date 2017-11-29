import torch
from dataio.reader.reader_base import ReaderBase
from preprocessor import SimplePreprocessor
from vocab import Vocab

class JsonSaliencePretrainReader():
    def __init__(self):
        #self.register_data("data_")
        self.embeddings = []
        self.labels = []

    def read(self, json_data):
        for datum in json_data:
            self.embeddings.append(datum["embedding"])
            self.labels.append(datum["label"])

    def finish(self):

        emb_size = len(self.embeddings[0])
        data_size = len(self.embeddings)
        embeddings = torch.FloatTensor(self.embeddings)
        labels = torch.LongTensor(self.labels)
        self.embeddings = []
        self.labels = []
        return (embeddings, labels)
