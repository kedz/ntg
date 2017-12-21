import os
import pkg_resources
from collections import OrderedDict


def english_wikipedia_frequencies():
    path = pkg_resources.resource_filename(
        'ntp.datasets.word_list', 
        os.path.join('data', 'enwiki_vocab_min200.txt'))
    word_freqs = OrderedDict()
    with open(path, "r") as fp:
        for line in fp:
            word, freq = line.strip().split()
            word_freqs[word] = int(freq)
    return word_freqs
