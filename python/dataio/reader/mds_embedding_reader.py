import torch

class MDSEmbeddingReader():
    def __init__(self):
        self.document_set_ids = []
        self.document_ids = []
        self.sentence_ids = []
        self.texts = []
        self.labels = []
        self.lengths = []
        self.embeddings = []

    def read(self, json_data):
        docset_ids = []
        doc_ids = []
        sent_ids = []
        texts = []
        labels = []
        embeddings = []
        for datum in json_data:
            docset_ids.append(datum["docset_id"])
            doc_ids.append(datum["doc_id"])
            sent_ids.append(int(datum["sentence_id"]))
            texts.append(datum["text"])
            labels.append(datum["label"])
            embeddings.append(datum["embedding"])
        self.lengths.append(len(labels))
        self.labels.append(torch.LongTensor(labels))
        self.embeddings.append(torch.FloatTensor(embeddings))
        self.texts.append(texts)
        self.sentence_ids.append(torch.LongTensor(sent_ids))

    def finish(self):
        emb_size = len(self.embeddings[0][0])
        max_docset_size = max(self.lengths)
        data_size = len(self.lengths)
        embeddings = torch.FloatTensor(data_size, max_docset_size, emb_size)
        embeddings.zero_()
        labels = torch.LongTensor(data_size, max_docset_size).fill_(-1)
        sent_ids = torch.LongTensor(data_size, max_docset_size).zero_()
        for i, emb in enumerate(self.embeddings):
            embeddings[i][:emb.size(0)].copy_(emb)
            labels[i][:emb.size(0)].copy_(self.labels[i])
            sent_ids[i][:emb.size(0)].copy_(self.sentence_ids[i])

        return (self.document_set_ids, self.document_ids, sent_ids, self.texts,
                embeddings, labels, torch.LongTensor(self.lengths))

