import dataio.tsv_reader
from dataset import SequencePredictionDataset

def read_sequence_data(path, pred_sequence_readers, 
                       input_sequence_readers=None,
                       skip_header=True, collect_stats=True, batch_size=1,
                       gpu=-1):

    if not isinstance(pred_sequence_readers, (list, tuple)):
        pred_sequence_readers = [pred_sequence_readers]

    if input_sequence_readers is None:
        input_sequence_readers = []
    elif not isinstance(input_sequence_readers, (list, tuple)):
        input_sequence_readers = [input_sequence_readers]

    readers = list(pred_sequence_readers) + list(input_sequence_readers)

    if collect_stats:
        dataio.tsv_reader.collect_tsv_stats(
            path, readers, skip_header=skip_header, diagnostics=True)

    data = dataio.tsv_reader.apply_tsv_readers(
        path, readers, skip_header=skip_header)
    
    pred_seq_data = data[:len(pred_sequence_readers)]

    decoder_inputs = []
    targets = []
    target_length = pred_seq_data[0][2]

    for input, output, length in pred_seq_data:
        decoder_inputs.append(input)
        targets.append(output)
    
    return SequencePredictionDataset(
        decoder_inputs, targets, target_length,
        batch_size=batch_size, gpu=gpu)

