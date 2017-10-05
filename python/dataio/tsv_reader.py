from text_preprocessor import VocabPreprocessor
from data_structs import Vocab


def collect_tsv_stats(path, readers, skip_header=True, diagnostics=True):

    with open(path, "r") as fp:
        if skip_header:
            fp.readline()
        for line in fp:
            data = line.split("\t")
            for reader in readers:
                reader.collect_stats(data)

    for reader in readers:
        reader.freeze_vocab()

    if diagnostics:
        for reader in readers:
            print(reader.info())

def apply_tsv_readers(path, readers, skip_header=True):

    with open(path, "r") as fp:
        if skip_header:
            fp.readline()
        for line in fp:
            data = line.split("\t")
            for reader in readers:
                reader.read(data)

    all_reader_data = []
    for reader in readers:
        all_reader_data.append(reader.finish())

    return all_reader_data


def read_vocabs_from_tsv(path, fields, vocab_args=None, skip_header=True):

    if not isinstance(fields, (tuple, list)):
        fields = [fields]
    for field in fields:
        if not isinstance(field, int) or field < 0:
            raise Exception("arg 1 (fields) must be int or list of ints.")

    if vocab_args is None:
        vocab_args = [dict() for _ in fields]
    elif isinstance(vocab_args, dict):
        vocab_args = [vocab_args for _ in fields]
    elif isinstance(vocab_args, (tuple, list)):
        if not len(vocab_args) == len(fields):
            raise Exception("Not enough vocab_args for number of fields.")

    preprocessors = [VocabPreprocessor(field) for field in fields]

    with open(path, "r") as f:
        if skip_header == True:
            f.readline()
        for line in f:
            items = line.strip().split("\t")
            for pp in preprocessors:
                pp.preprocess(items)

    vocabs = [Vocab(pp, **args) for pp, args in zip(preprocessors, vocab_args)]

    if len(vocabs) == 1:
        return vocabs[0]
    else:
        return tuple(vocabs)
