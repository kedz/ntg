import torch
from dataio.dataset import LMDataset, Seq2SeqDataset

from dataio.sequence_reader import DiscreteSequenceReader

def read_lm_dataset(path, field, vocab, skip_header=True):

    reader = DiscreteSequenceReader(
        field, vocab, left_pad="_START_", right_pad="_STOP_",
        offset_io_pair=True)

    data = read_tsv_dataset(path, [reader], skip_header=skip_header)
    input, output, length = data[0]

    inputs = ((input, length, "input"),)
    targets = ((output, length, "target"),)
    sorting_length = length

    dataset = Dataset(inputs, targets, sorting_length)
    return dataset

def read_seq2seq_dataset(path, field_src, vocab_src, field_tgt, vocab_tgt, 
                         skip_header=True, batch_size=1, chunk_size=500,
                         gpu=-1):
    
    reader_src = DiscreteSequenceReader(
        field_src, vocab_src, left_pad=None, right_pad=None,
        offset_io_pair=False)
    reader_tgt = DiscreteSequenceReader(
        field_tgt, vocab_tgt, left_pad="_START_", right_pad="_STOP_",
        offset_io_pair=True)

    data = read_tsv_dataset(
        path, [reader_src, reader_tgt], skip_header=skip_header)
    source, source_length = data[0]
    target_in, target_out, target_length = data[1]

    encoder_inputs = ((source, "encoder_input"),) 
    decoder_inputs = ((target_in, "decoder_input"),)

    dataset = Seq2SeqDataset(
        encoder_inputs, source_length,
        decoder_inputs, target_out, target_length,
        batch_size=batch_size,
        chunk_size=chunk_size,
        gpu=gpu)

    return dataset


    dataset = Dataset(inputs, targets, sorting_length)
    return dataset

def read_seq2seq_rs_dataset(path, field_src, vocab_src, field_tgt, vocab_tgt,
                            max_steps=50, skip_header=True,
                            batch_size=1, chunk_size=500, gpu=-1):

    reader_src = DiscreteSequenceReader(
        field_src, vocab_src, left_pad=None, right_pad=None,
        offset_io_pair=False)
    reader_tgt = DiscreteSequenceReader(
        field_tgt, vocab_tgt, left_pad="_START_", right_pad="_STOP_",
        offset_io_pair=True)

    data = read_tsv_dataset(
        path, [reader_src, reader_tgt], skip_header=skip_header)
    source, source_length = data[0]
    target_in, target_out, target_length = data[1]

    remaining_steps = target_out.new(target_out.size()).fill_(0)
    
    for i in range(remaining_steps.size(0)):
        max = target_length[i]
        torch.arange(max, 0, step=-1, out=remaining_steps[i,:max])

    mask = remaining_steps.ge(max_steps)
    remaining_steps.masked_fill_(mask, max_steps - 1)

    encoder_inputs = ((source, "encoder_input"),) 
    decoder_inputs = ((target_in, "decoder_input"),
                      (remaining_steps, "remaining_steps"))

    dataset = Seq2SeqDataset(
        encoder_inputs, source_length,
        decoder_inputs, target_out, target_length,
        batch_size=batch_size, chunk_size=chunk_size, gpu=gpu)
    return dataset

def read_seq2seq_ts_dataset(path, field_src, vocab_src, field_tgt, vocab_tgt,
                            max_steps=50, skip_header=True,
                            batch_size=1, chunk_size=500, gpu=-1):

    reader_src = DiscreteSequenceReader(
        field_src, vocab_src, left_pad=None, right_pad=None,
        offset_io_pair=False)
    reader_tgt = DiscreteSequenceReader(
        field_tgt, vocab_tgt, left_pad="_START_", right_pad="_STOP_",
        offset_io_pair=True)

    data = read_tsv_dataset(
        path, [reader_src, reader_tgt], skip_header=skip_header)
    source, source_length = data[0]
    target_in, target_out, target_length = data[1]

    remaining_steps = target_out.new(target_out.size()).fill_(0)
    
    for i in range(remaining_steps.size(0)):
        total_steps = min(target_length[i], max_steps - 1)
        remaining_steps[i,:target_length[i]].fill_(total_steps)
    mask = remaining_steps.gt(max_steps)
    remaining_steps.masked_fill_(mask, max_steps)

    encoder_inputs = ((source, "encoder_input"),) 
    decoder_inputs = ((target_in, "decoder_input"),
                      (remaining_steps, "total_steps"))

    dataset = Seq2SeqDataset(
        encoder_inputs, source_length,
        decoder_inputs, target_out, target_length,
        batch_size=batch_size, chunk_size=chunk_size, gpu=gpu)
    return dataset



def read_tsv_dataset(path, readers, skip_header=True):
    with open(path, "r") as fp:
        if skip_header:
            fp.readline()
        for line in fp:
            items = line.split("\t")
            for reader in readers:
                reader.read(items)

    data = [reader.finish() for reader in readers]
    return data


def read_dataset_from_tsv(path, vocab, skip_header=True, 
                          start_token="_START_", stop_token="_STOP_"):

    inputs = []
    outputs = []
    lengths = []

    start = vocab.index(start_token)
    stop = vocab.index(stop_token)

    with open(path, "r") as f:
        if skip_header:
            f.readline()
        for line in f:
            items = line.strip().split("\t")
            seq = vocab.preprocess_lookup(items)
            inputs.append([start] + seq)
            outputs.append(seq + [stop])
            lengths.append(len(seq) + 1)


    data_size = len(inputs)
    max_len = max(lengths)

    for i in range(data_size):
        if lengths[i] < max_len:
            inputs[i] += [0] * (max_len - lengths[i])
            outputs[i] += [0] * (max_len - lengths[i])

    inputs = torch.LongTensor(inputs)
    outputs = torch.LongTensor(outputs)
    lengths = torch.LongTensor(lengths)

    return LMDataset(inputs, outputs, lengths)
