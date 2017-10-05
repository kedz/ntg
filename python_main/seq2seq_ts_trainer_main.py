import os
import argparse
import json

import dataio
from model_builder import Seq2SeqRNNBuilder
import criterion
import trainer

import random
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train seq2seq.")

    parser.add_argument(
        "--train", required=True)
    parser.add_argument(
        "--valid", required=True)

    parser.add_argument(
        "--rnn-type", required=False, type=str, default="gru",
        choices=["rnn", "gru", "lstm"])
    parser.add_argument(
        "--num-layers", default=1, type=int, required=False)
    parser.add_argument(
        "--bidirectional", default=1, type=int, choices=[0,1])
    parser.add_argument(
        "--embedding-size", required=False, type=int, default=300)
    parser.add_argument(
        "--hidden-size", required=False, type=int, default=300)
    parser.add_argument(
        "--bridge-type", required=False, type=str, default="linear-relu",
        choices=["average", "linear-relu"])
    parser.add_argument(
        "--attention-type", required=False, choices=["none", "dot",],
        default="dot")

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
        "--src-tgt-fields", required=False, default=(0, 1,), type=int,
        nargs=2)
    parser.add_argument(
        "--enc-vocab-size", default=40000, type=int, required=False)
    parser.add_argument(
        "--dec-vocab-size", default=40000, type=int, required=False)
    parser.add_argument(
        "--max-steps", default=50, type=int, required=False)
    parser.add_argument(
        "--step-embedding-size", default=100, type=int, required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None, required=False, type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)


    field_src, field_tgt = args.src_tgt_fields
    vocab_args_src = {"special_tokens": [], 
                      "unknown_token": "_UNK_",
                      "top_k": args.enc_vocab_size}
    vocab_args_tgt = {"special_tokens": ["_START_", "_STOP_"], 
                      "unknown_token": "_UNK_", 
                      "top_k": args.dec_vocab_size}

    vocab_src, vocab_tgt = dataio.read_vocabs_from_tsv(
        args.train,
        [field_src, field_tgt],
        vocab_args=[vocab_args_src, vocab_args_tgt])

    data_train = dataio.read_seq2seq_ts_dataset(
        args.train, field_src, vocab_src, field_tgt, vocab_tgt,
        max_steps=args.max_steps, skip_header=True,
        batch_size=args.batch_size, gpu=args.gpu)
    data_valid = dataio.read_seq2seq_ts_dataset(
        args.valid, field_src, vocab_src, field_tgt, vocab_tgt,
        max_steps=args.max_steps, skip_header=True,
        batch_size=args.batch_size, gpu=args.gpu)

    builder = Seq2SeqRNNBuilder()
    builder.add_encoder(
        vocab_src.size, args.embedding_size, args.rnn_type, args.hidden_size,
        args.num_layers, args.bidirectional == 1)
    builder.add_decoder(
        [vocab_tgt.size, args.max_steps], 
        [args.embedding_size, args.step_embedding_size],
        args.rnn_type, args.hidden_size, args.num_layers, 
        vocab_tgt.size,
        vocab_tgt.index("_START_"),
        vocab_tgt.index("_STOP_"),
        args.attention_type)
    builder.add_bridge(
        args.bridge_type, args.bidirectional == 1, args.hidden_size,
        args.num_layers)
    model = builder.finish_rnn_ts()

    model.set_meta("model_type", "seq2seq_rnn_ts")
    model.set_meta("encoder_vocab", vocab_src)
    model.set_meta("decoder_vocab", vocab_tgt)
    model.set_meta("max_steps", args.max_steps)
   
    #import types

    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)
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
