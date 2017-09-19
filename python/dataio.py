from text_preprocessor import VocabPreprocessor
from data_structs import Vocab, LengthVocab, Seq2SeqDataset
import torch

START_DECODE_SYMBOL = "_START_"
STOP_DECODE_SYMBOL = "_STOP_"
UNKNOWN_TOKEN_SYMBOL = "_UNK_"


def load_data(train_path, valid_path, src_field=0, tgt_field=1, 
              length_mode=None, len_field=2, enc_vocab_size=100000,
              dec_vocab_size=100000):

    if length_mode is not None and length_mode != "none":
        vocab_src, vocab_tgt, vocab_len = extract_vocab_from_tsv(
            train_path, source_field=src_field, target_field=tgt_field,
            length_field=len_field, enc_vocab_size=enc_vocab_size,
            dec_vocab_size=dec_vocab_size)

        print("src vocab: {} tokens.".format(vocab_src.size))
        print("tgt vocab: {} tokens.".format(vocab_tgt.size))
        print("len vocab: {} tokens.".format(vocab_len.size))
        train_data = extract_dataset_from_tsv(
            train_path, vocab_src, vocab_tgt, vocab_len=vocab_len)
        valid_data = extract_dataset_from_tsv(
            valid_path, vocab_src, vocab_tgt, vocab_len=vocab_len)

        return (vocab_src, vocab_tgt, vocab_len), (train_data, valid_data)

    else:
         
        vocab_src, vocab_tgt = extract_vocab_from_tsv(
            train_path, source_field=src_field, target_field=tgt_field,
            enc_vocab_size=enc_vocab_size, dec_vocab_size=dec_vocab_size)
        print("src vocab: {} tokens.".format(vocab_src.size))
        print("tgt vocab: {} tokens.".format(vocab_tgt.size))
        train_data = extract_dataset_from_tsv(
            train_path, vocab_src, vocab_tgt)
        valid_data = extract_dataset_from_tsv(
            valid_path, vocab_src, vocab_tgt)
        
        return (vocab_src, vocab_tgt), (train_data, valid_data)

    exit()


def extract_vocab_from_tsv(path, source_field=0, target_field=1, 
                           length_field=None,
                           skip_header=True, enc_vocab_size=100000,
                           dec_vocab_size=100000):

    vp_src = VocabPreprocessor(source_field)
    vp_tgt = VocabPreprocessor(target_field)
    if length_field is not None:
        vp_len = VocabPreprocessor(length_field, replace_digits=False)
    
    with open(path, "r") as f:
        if skip_header == True:
            f.readline()
        for line in f:
            items = line.strip().split("\t")
            vp_src.preprocess(items)
            vp_tgt.preprocess(items)
            if length_field is not None:
                vp_len.preprocess(items)

    vocab_src = Vocab(
        vp_src, 
        top_k=enc_vocab_size,
        unknown_token=UNKNOWN_TOKEN_SYMBOL)
    vocab_tgt = Vocab(
        vp_tgt, 
        special_tokens=[START_DECODE_SYMBOL, STOP_DECODE_SYMBOL], 
        unknown_token=UNKNOWN_TOKEN_SYMBOL,
        top_k=dec_vocab_size)
    
    if length_field is not None:
        vocab_len = LengthVocab(
            vp_len, at_least=100, unknown_token=UNKNOWN_TOKEN_SYMBOL)
        return vocab_src, vocab_tgt, vocab_len

    else:
        return vocab_src, vocab_tgt


def extract_dataset_from_tsv(path, vocab_src, vocab_tgt, vocab_len=None, 
                             skip_header=True):

    sources = []
    targets_in = []
    targets_out = []
    source_lengths = []
    target_lengths = []
    target_len_toks = []

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
            if vocab_len:
                target_len_toks.append(vocab_len.preprocess_lookup(items))

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

    if vocab_len:
        target_len_toks = torch.LongTensor(target_len_toks)
    else:
        target_len_toks = None

    dataset = Seq2SeqDataset(
        sources, source_lengths, targets_in, targets_out, target_lengths,
        target_len_toks)

    return dataset
