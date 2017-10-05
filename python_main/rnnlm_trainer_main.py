import os
import argparse
import json

import dataio
import models
import criterion
import trainer

import random
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an rnn language model.")

    parser.add_argument(
        "--train", required=True)
    parser.add_argument(
        "--valid", required=True)

    parser.add_argument(
        "--rnn-type", required=False, type=str, default="rnn",
        choices=["rnn", "gru", "lstm"])
    parser.add_argument(
        "--num-layers", default=1, type=int, required=False)
    parser.add_argument(
        "--embedding-size", required=False, type=int, default=300)
    parser.add_argument(
        "--hidden-size", required=False, type=int, default=300)

    parser.add_argument(
        "--gpu", default=-1, type=int, required=False)
    parser.add_argument(
        "--epochs", default=25, type=int, required=False)
    parser.add_argument(
        "--seed", default=83419234, type=int, required=False)
    
    parser.add_argument(
        "--optimizer", default="adagrad", type=str, required=False,
        choices=["sgd", "adagrad", "adadelta", "adam"])
    parser.add_argument(
        "--lr", required=False, default=.001, type=float)
    parser.add_argument(
        "--batch-size", default=16, type=int, required=False)
    parser.add_argument(
        "--dropout", default=0.0, required=False, type=float)

    parser.add_argument(
        "--input-field", required=False, default=0, type=int)
    parser.add_argument(
        "--vocab-size", default=40000, type=int, required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None,required=False, type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    vocab_args = {"special_tokens": ["_START_", "_STOP_"], 
                  "unknown_token": "_UNK_", 
                  "top_k": args.vocab_size}

    vocab = dataio.read_vocabs_from_tsv(
        args.train,
        args.input_field,
        vocab_args=vocab_args)

    data_train = dataio.read_lm_dataset(args.train, args.input_field, vocab)
    data_valid = dataio.read_lm_dataset(args.valid, args.input_field, vocab)

    data_train.set_batch_size(args.batch_size)
    data_valid.set_batch_size(args.batch_size)

    model = models.RNNLM(
        args.rnn_type,
        vocab,
        args.embedding_size,
        args.hidden_size,
        args.num_layers)

    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)
            data_train.set_gpu(args.gpu)
            data_valid.set_gpu(args.gpu)
        model = model.cuda(args.gpu)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise Exception("Unkown optimizer: {}".format(args.optimizer))

    if args.save_model is not None:
        d = os.path.dirname(args.save_model)
        if d != "" and not os.path.exists(d):
            os.makedirs(d)

    crit = criterion.SequenceCrossEntropy(model, optimizer)
    results = trainer.minimize_criterion(
        crit, data_train, data_valid, args.epochs,
        save_best_model=args.save_model)

    best_valid_loss = min(results["valid_nll"])
    best_epoch = results["valid_nll"].index(best_valid_loss) + 1
    print("\nBest epoch: {}".format(best_epoch))
    print("Best validation nll: {}".format(best_valid_loss))

    if args.save_results is not None:
        d = os.path.dirname(args.save_results)
        if d != "" and not os.path.exists(d):
            os.makedirs(d)

        print("Writing results json to {} ...".format(args.save_results))
        results["parameters"] = vars(args)
        with open(args.save_results, "w") as fp:
            json.dump(results, fp)
