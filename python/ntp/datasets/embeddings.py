from .util import get_data_dir, download_url_to_file, \
    download_google_drive_to_buffer
from ..dataio import Vocabulary
import os
import urllib.request
import zipfile
import gzip
import sys
import torch
import struct


def copy_from_pretrained(name, size, dest, vocab, verbose=False):

    replace_count = 0
    if name.startswith("glove"):
        emb_iter = get_glove_iter(name, size)

        for word, lazy_embedding in emb_iter:
            idx = vocab[word] 
            if idx != vocab.unknown_index:
                dest[idx].copy_(lazy_embedding())
                replace_count += 1
    elif name == "word2vec":
        vocab_and_emb = torch.load(get_word2vec_data_path(size))
        w2v_vocab = vocab_and_emb["vocab"]
        w2v_embeddings = vocab_and_emb["embeddings"]

        for word in vocab:
            index = vocab[word]
            if w2v_vocab.contains(word):
                dest[index].copy_(w2v_embeddings[w2v_vocab[word]])
            replace_count += 1
    else:
        raise Exception("Invalid name argument.")
    if verbose:
        print("Replaced {} of {} words ({:0.3f}%).".format(
            replace_count, vocab.size, replace_count / vocab.size * 100))

def get_glove_iter(name, size):
    path = get_glove_data_path(name, size)
    filename = "{}.{}d.txt".format(name, size)
    with zipfile.ZipFile(path) as zfp:

        if not filename in zfp.namelist():
            raise Exception(
                "Bad file: {}\nTry deleting and run again.".format(path))
        with zfp.open(filename) as fp:
            for line in fp:
                items = line.split()
                word = items[0].decode("utf8")
                def lazy_embedding():
                    return torch.FloatTensor([float(x) for x in items[1:]])
                yield word, lazy_embedding

def get_glove_embeddings(name, size, filter_support=None):
    path = get_glove_data_path(name, size)
    filename = "{}.{}d.txt".format(name, size)
    with zipfile.ZipFile(path) as zfp:
        if not filename in zfp.namelist():
            raise Exception(
                "Bad file: {}\nTry deleting and run again.".format(path))
        with zfp.open(filename) as fp:
            words = []
            data = []
            for line in fp:
                items = line.split()
                word = items[0].decode("utf8")

                if filter_support is not None:
                    if word not in filter_support:
                        continue

                words.append(word)
                data.append([float(x) for x in items[1:]])
            vocab = Vocabulary(special_tokens=words, zero_indexing=True)
            vocab.freeze()
            return vocab, torch.FloatTensor(data)

def get_glove_data_path(name, size):
    if name == "glove.840B" and size == 300:
        filename = "{}.{}d.zip".format(name, size)
        path = os.path.join(get_data_dir(), "glove", filename)
    elif name == "glove.6B" and size in [50, 100, 200, 300]:
        filename = "glove.6B.zip"
        path = os.path.join(get_data_dir(), "glove", filename)
    else:
        raise Exception()
    
    if not os.path.exists(path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        url = "http://nlp.stanford.edu/data/{}".format(filename)   
        download_url_to_file(url, path)

    return path

def get_word2vec_data_path(size):
    if size != 300:
        raise Exception("word2vec only available for 300 dimensional vectors.")

    path = os.path.join(get_data_dir(), "word2vec", "word2vec.300.th.bin")

    if not os.path.exists(path):

        google_drive_id = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"
        buffer = download_google_drive_to_buffer(google_drive_id)
        
        float_size = 4
        words = []
        embeddings = []
        with gzip.GzipFile(fileobj=buffer, mode="rb") as fp:
            header = fp.readline()
            vocab_size, emb_size = map(int, header.split())

            for line in range(vocab_size):
                word = []
                while True:
                    ch = fp.read(1)
                    if ch == b' ':
                        word = b''.join(word).decode("utf8")
                        break
                    if ch != b'\n':
                        word.append(ch)   
                words.append(word)
                float_str = fp.read(float_size * emb_size)
                vec = struct.unpack("{}f".format(emb_size), float_str)
                embeddings.append(vec)

        embeddings = torch.FloatTensor(embeddings)
        vocab = Vocabulary(zero_indexing=True, special_tokens=words)
        vocab.freeze()

        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        torch.save({"vocab": vocab, "embeddings": embeddings}, path)

    if not os.path.exists(path):
        raise Exception("Could not download word2vec embeddings!")

    return path
