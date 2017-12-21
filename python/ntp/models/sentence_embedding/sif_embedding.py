import torch
from ntp.datasets.embeddings import get_glove_embeddings
from ntp.datasets import word_list
from sklearn.decomposition import TruncatedSVD


class SIFEmbedding(object):

    @staticmethod
    def from_pretrained(embedding_name="glove.6B", embedding_size=300,
                        word_frequencies="english_wikipedia", alpha=1e-3): 

        if word_frequencies == "english_wikipedia":
            word_freq = word_list.english_wikipedia_frequencies()
        else:
            raise Exception("No other word frequency lists provided yet.")

        vocab, embeddings = get_glove_embeddings(
            embedding_name, embedding_size, 
            filter_support=set([w for w in word_freq.keys()]))
        freq = torch.LongTensor([word_freq[token] for token in vocab])
        return SIFEmbedding(vocab, embeddings, freq, alpha=alpha)

    def __init__(self, vocab, embeddings, frequencies, alpha=1e-3):
        self.vocab = vocab
        self.frequencies = frequencies
        self.alpha = alpha

        Z = float(self.frequencies.sum())
        probs = self.frequencies.float() / float(Z)
        self.weights = self.alpha / (self.alpha + probs)
   
        emb_size = embeddings.size(1)
        self.weighted_word_embeddings = self.weights.view(-1, 1, 1).bmm(
            embeddings.view(-1, 1, emb_size)).view(-1, emb_size)
        self.pc = None

    def fit_principle_component(self, sentences):
        sent_emb = self.embed_sentences(sentences, remove_pc=False)
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(sent_emb)
        self.pc = torch.from_numpy(svd.components_[0]).contiguous()

    def embed_sentences(self, sentences, remove_pc=True):
        sent_embeddings = self.weighted_word_embeddings.new(
            len(sentences), self.weighted_word_embeddings.size(1)).fill_(0)

        for i, sentence in enumerate(sentences):
            Z = 0
            for token in sentence:    
                if self.vocab.contains(token):
                    Z += 1
                    sent_embeddings[i].add_(
                        self.weighted_word_embeddings[self.vocab[token]])
            if Z > 0:
                sent_embeddings[i].div_(Z)

        if remove_pc and self.pc is not None:
            magnitude = sent_embeddings.matmul(self.pc)
            pc_proj = magnitude.view(-1, 1).mm(self.pc.view(1, -1))
            return sent_embeddings - pc_proj

        else:
            return sent_embeddings
