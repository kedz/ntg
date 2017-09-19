import argparse
import torch
from dataio import extract_dataset_from_tsv
from sklearn.linear_model import Ridge, Lasso
import numpy as np


import random
random.seed(12345)


def extract_data(model, dataset):

    
    dims = model.encoder.embeddings_.weight.size(1)
    activations = torch.zeros(dataset.size, dims)
    lengths = torch.zeros(dataset.size)

    position = 0
    for i, batch in enumerate(dataset.iter_batch(), 1):
        _, enc_out = model.encoder(batch.source, batch.source_length)

        batch_size = batch.source.size(0)
        activations[position:position + batch_size].copy_(enc_out.data[0])
        lengths[position:position + batch_size].copy_(batch.target_length)

        position += batch_size
   
    return activations.numpy(), lengths.numpy()



if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--train", required=True, type=str)
    parser.add_argument("--valid", required=True, type=str)
    parser.add_argument("--test", required=True, type=str)
    parser.add_argument("--batch-size", required=False, type=int, default=4)

    args = parser.parse_args()

    model = torch.load(args.model)

    vocab_src = model.encoder.vocab_
    vocab_tgt = model.decoder.vocab

    data_train = extract_dataset_from_tsv(args.train, vocab_src, vocab_tgt)
    data_valid = extract_dataset_from_tsv(args.valid, vocab_src, vocab_tgt)
    data_test = extract_dataset_from_tsv(args.test, vocab_src, vocab_tgt)

    data_train.set_batch_size(args.batch_size)
    data_train.set_gpu(0)
    data_valid.set_batch_size(args.batch_size)
    data_valid.set_gpu(0)
    data_test.set_batch_size(args.batch_size)
    data_test.set_gpu(0)
    

    model.eval()

    model.encoder.rnn_.flatten_parameters()
    print("Reading training data.")
    act_train, len_train = extract_data(model, data_train)
    print("Reading validation data.")
    act_valid, len_valid = extract_data(model, data_valid)
    
    reg_model = Ridge(alpha=0, max_iter=10000)
    reg_model.fit(act_train, len_train)
    print("alpha=0")
    print(reg_model.score(act_train, len_train))
    print(reg_model.score(act_valid, len_valid))

    reg_model = Lasso(alpha=1.0, max_iter=10000)
    reg_model.fit(act_train, len_train)
    print("alpha=1.0")
    print(reg_model.score(act_train, len_train))
    print(reg_model.score(act_valid, len_valid))
    print(np.count_nonzero(reg_model.coef_))

    reg_model = Lasso(alpha=.1, max_iter=10000)
    reg_model.fit(act_train, len_train)
    print("alpha=.1")
    print(reg_model.score(act_train, len_train))
    print(reg_model.score(act_valid, len_valid))
    print(np.count_nonzero(reg_model.coef_))

    reg_model = Lasso(alpha=.01, max_iter=10000)
    reg_model.fit(act_train, len_train)
    print("alpha=.01")
    print(reg_model.score(act_train, len_train))
    print(reg_model.score(act_valid, len_valid))
    print(np.count_nonzero(reg_model.coef_))

    reg_model = Lasso(alpha=.001, max_iter=10000)
    reg_model.fit(act_train, len_train)
    print("alpha=.001")
    print(reg_model.score(act_train, len_train))
    print(reg_model.score(act_valid, len_valid))
    print(np.count_nonzero(reg_model.coef_))

    reg_model = Lasso(alpha=.0001, max_iter=10000)
    reg_model.fit(act_train, len_train)
    print("alpha=.0001")
    print(reg_model.score(act_train, len_train))
    print(reg_model.score(act_valid, len_valid))
    print(np.count_nonzero(reg_model.coef_))



