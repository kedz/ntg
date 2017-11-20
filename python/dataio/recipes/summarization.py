import dataio.json_reader
from dataset import MDSDataset
from dataset.SimpleClassificationDataset import SimpleClassificationDataset

def read_mds_embedding_data(path, mds_reader,
                            batch_size=1, gpu=-1):

    data = dataio.json_reader.apply_json_readers(path, [mds_reader])
    ds_ids, did, sid, text, emb, label, length = data[0]

    return MDSDataset(emb, length, label, text, batch_size=batch_size, gpu=gpu)

def read_salience_pretraining_data(path, reader, batch_size=1, gpu=-1):
    (features, labels), = dataio.json_reader.apply_json_readers(
        path, [reader])
    return SimpleClassificationDataset(features, labels, 
        batch_size=batch_size, gpu=gpu)
