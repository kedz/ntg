from .util import get_data_dir, download_url_to_file
from ..dataio import Vocabulary
import os
import urllib.request
import zipfile
import sys
import torch


def copy_from_pretrained(name, size, dest, vocab, verbose=False):

    if name.startswith("glove"):
        emb_iter = get_glove_iter(name, size)
    else:
        raise Exception("Invalid name argument.")

    replace_count = 0
    for word, lazy_embedding in emb_iter:
        idx = vocab[word] 
        if idx != vocab.unknown_index:
            dest[idx].copy_(lazy_embedding())
            replace_count += 1
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
