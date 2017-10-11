import dataio.tsv_reader
from dataset import SequencePredictionDataset

def read_sequence_data(path, pred_sequence_readers, 
                       input_sequence_readers=None,
                       feature_readers=None,
                       skip_header=True, collect_stats=True, batch_size=1,
                       gpu=-1):

    if not isinstance(pred_sequence_readers, (list, tuple)):
        pred_sequence_readers = [pred_sequence_readers]

    if input_sequence_readers is None:
        input_sequence_readers = []
    elif not isinstance(input_sequence_readers, (list, tuple)):
        input_sequence_readers = [input_sequence_readers]
    
    if feature_readers is None:
        feature_readers = []
    elif not isinstance(feature_readers, (list, tuple)):
        feature_readers = [feature_readers]

    readers = list(pred_sequence_readers) + list(input_sequence_readers) \
        + list(feature_readers)

    if collect_stats:
        dataio.tsv_reader.collect_tsv_stats(
            path, readers, skip_header=skip_header, diagnostics=True)

    data = dataio.tsv_reader.apply_tsv_readers(
        path, readers, skip_header=skip_header)

    num_pred_seq = len(pred_sequence_readers)
    num_inp_seq = len(input_sequence_readers)
    num_ftr = len(feature_readers)

    pred_seq_data = data[:num_pred_seq]
    input_seq_data = data[num_pred_seq:num_pred_seq + num_inp_seq]
    ftr_seq_data = data[num_pred_seq + num_inp_seq:]

    decoder_features = []
    encoder_inputs = []
    decoder_inputs = []
    targets = []
    target_length = pred_seq_data[0][2]

    for input, length in input_seq_data:
        encoder_inputs.append(input)
    if len(input_seq_data) > 0:
        encoder_length = input_seq_data[0][1]
    else:
        encoder_length = None

    for input, output, length in pred_seq_data:
        decoder_inputs.append(input)
        targets.append(output)
    
    for feature, in ftr_seq_data:
        decoder_features.append(feature)

    return SequencePredictionDataset(
        decoder_inputs, 
        targets, 
        target_length,
        encoder_inputs=encoder_inputs,
        encoder_length=encoder_length,
        decoder_features=decoder_features,
        batch_size=batch_size, 
        gpu=gpu)
