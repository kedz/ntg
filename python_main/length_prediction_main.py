import os
import torch
import dataio
from sklearn.linear_model import LinearRegression, Lasso
import numpy as np
import argparse


def get_length(m):
    lengths = m.data.gt(0).sum(1)
    return lengths

def encode_data(model, dataset):

    state_size = model.encoder.num_layers
    state_size *= (2 if model.encoder.bidirectional else 1)
    state_size *= model.encoder.hidden_size
    encoded_states = torch.FloatTensor(dataset.size, state_size).fill_(0)
    target_lengths = torch.FloatTensor(dataset.size).fill_(0)
    
    print(model.encoder.num_layers)
    print(model.encoder.bidirectional)
    print(model.encoder.hidden_size)

    position = 0
    for step, batch in enumerate(dataset.iter_batch(), 1):
        batch_size = batch.encoder_input.size(0)
        _, state = model.encode_batch(batch)
        encoded_states[position:position + batch_size].copy_(
            state.data.permute(1,0,2).contiguous().view(batch_size, -1))
        

        target_lengths[position:position + batch_size].copy_(
            get_length(batch.target))
        position += batch_size 
    
    return encoded_states, target_lengths
    

def load_data(model, path, batch_size=32, gpu=-1):
    vocab_src = model.encoder_vocab
    vocab_tgt = model.decoder_vocab

    data = dataio.read_seq2seq_dataset(
        path, 0, vocab_src, 1, vocab_tgt,
        skip_header=True)
    data.set_batch_size(args.batch_size)
    if gpu > -1:
        data.set_gpu(gpu)

    return encode_data(model, data)


def cache_data(data, path):
    dirname = os.path.dirname(path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    torch.save(data, path)

def load_model(path, gpu=-1):
    # load model to cpu regardless of source.
    model = torch.load(path, map_location=lambda storage, loc: storage)
    if gpu > -1:
        model.cuda(gpu)
    model.eval()

    return model

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--train", required=True, type=str)
    parser.add_argument(
        "--cache-train", required=False, type=str, default=None)
    parser.add_argument("--valid", required=True, type=str)
    parser.add_argument(
        "--cache-valid", required=False, type=str, default=None)
#    parser.add_argument("--test", required=True, type=str)
    parser.add_argument("--batch-size", required=False, type=int, default=32)

    args = parser.parse_args()
    overwrite = False
    model = None


    if os.path.exists(args.cache_train) and overwrite is False:
        X_train, Y_train = torch.load(args.cache_train)
    else:
        if model is None:
            model = load_model(args.model, gpu=0)
        X_train, Y_train = load_data(
            model, args.train, batch_size=args.batch_size, gpu=0)
        if args.cache_train is not None:
            cache_data((X_train, Y_train), args.cache_train)

    X_train = X_train.numpy()
    Y_train = Y_train.view(-1,1).numpy()

    if os.path.exists(args.cache_valid) and overwrite is False:
        X_valid, Y_valid = torch.load(args.cache_valid)
    else:
        if model is None:
            model = load_model(args.model, gpu=0)
        X_valid, Y_valid = load_data(
            model, args.valid, batch_size=args.batch_size, gpu=0)
        if args.cache_valid is not None:
            cache_data((X_valid, Y_valid), args.cache_valid)

    X_valid = X_valid.numpy()
    Y_valid = Y_valid.view(-1,1).numpy()

    regressor = LinearRegression()
    print(regressor.fit(X_train, Y_train).score(X_train, Y_train))
    print(regressor.score(X_valid, Y_valid))
    print(np.abs(regressor.predict(X_train).round() - Y_train).mean())
    print(np.abs(regressor.predict(X_valid).round() - Y_valid).mean())

    print("")
    regressor = Lasso(alpha=.005, max_iter=10000)
    print(regressor.fit(X_train, Y_train).score(X_train, Y_train))
    print(regressor.score(X_valid, Y_valid))
    print(np.count_nonzero(regressor.coef_))
    print(np.abs(regressor.predict(X_train).round() - Y_train).mean())
    print(np.abs(regressor.predict(X_valid).round() - Y_valid).mean())
   

    






