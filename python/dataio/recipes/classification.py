import dataio.tsv_reader
from dataset import SequenceClassificationDataset

def read_labeled_sequence_data(path, sequence_reader, label_reader, 
                               skip_header=True, collect_stats=True,
                               batch_size=1, gpu=-1):

    readers = [sequence_reader, label_reader]

    if collect_stats:
        dataio.tsv_reader.collect_tsv_stats(
            path, readers, skip_header=skip_header,
            diagnostics=True)

    data = dataio.tsv_reader.apply_tsv_readers(
        path, readers, skip_header=skip_header)

    input, input_length = data[0]
    target, = data[1]

    return SequenceClassificationDataset(
        [input], input_length, [target], batch_size=batch_size, gpu=gpu)
