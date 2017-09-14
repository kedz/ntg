from text_preprocessor import VocabPreprocessor
from data_structs import Vocab, Seq2SeqDataset
import torch

START_DECODE_SYMBOL = "_START_"
STOP_DECODE_SYMBOL = "_STOP_"
UNKNOWN_TOKEN_SYMBOL = "_UNK_"


def extract_vocab_from_tsv(path, source_field=0, target_field=1, 
                           skip_header=True):

    vp_src = VocabPreprocessor(source_field)
    vp_tgt = VocabPreprocessor(target_field)
    
    with open(path, "r") as f:
        if skip_header == True:
            f.readline()
        for line in f:
            items = line.strip().split("\t")
            vp_src.preprocess(items)
            vp_tgt.preprocess(items)

    vocab_src = Vocab(vp_src)
    vocab_tgt = Vocab(
        vp_tgt, 
        special_tokens=[START_DECODE_SYMBOL, STOP_DECODE_SYMBOL], 
        unknown_token=UNKNOWN_TOKEN_SYMBOL)

    return vocab_src, vocab_tgt


def extract_dataset_from_tsv(path, vocab_src, vocab_tgt, skip_header=True):

    sources = []
    targets_in = []
    targets_out = []
    source_lengths = []
    target_lengths = []

    dec_start = vocab_tgt.index(START_DECODE_SYMBOL)
    dec_stop = vocab_tgt.index(STOP_DECODE_SYMBOL)

    with open(path, "r") as f:
        if skip_header:
            f.readline()
        for line in f:
            items = line.strip().split("\t")
            src = vocab_src.preprocess_lookup(items)
            tgt = vocab_tgt.preprocess_lookup(items)
            sources.append(src)
            source_lengths.append(len(src))
            targets_in.append([dec_start] + tgt)
            targets_out.append(tgt + [dec_stop])
            target_lengths.append(len(tgt) + 1)

    data_size = len(sources)
    max_len_src = max(source_lengths)
    max_len_tgt = max(target_lengths)

    for i in range(data_size):
        if source_lengths[i] < max_len_src:
            sources[i] += [0] * (max_len_src - source_lengths[i])

        if target_lengths[i] < max_len_tgt:
            targets_in[i] += [0] * (max_len_tgt - target_lengths[i])
            targets_out[i] += [0] * (max_len_tgt - target_lengths[i])

    sources = torch.LongTensor(sources)
    source_lengths = torch.LongTensor(source_lengths)
    
    targets_in = torch.LongTensor(targets_in)
    targets_out = torch.LongTensor(targets_out)
    target_lengths = torch.LongTensor(target_lengths)


    dataset = Seq2SeqDataset(
        sources, source_lengths, targets_in, targets_out, target_lengths)

    return dataset
