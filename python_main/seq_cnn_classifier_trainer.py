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
        "--embedding-size", required=False, type=int, default=300)
    parser.add_argument(
        "--filter-size", required=False, type=int, default=100)
    parser.add_argument(
        "--filter-widths", required=False, type=int, nargs="+", default=(1,))

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
        "--layer-norm", default=0, type=int, required=False, choices=[0,1])
    parser.add_argument(
        "--l2", default=0, type=float, required=False)

    parser.add_argument(
        "--input-field", required=False, default=1, type=int)
    parser.add_argument(
        "--target-field", required=False, default=0, type=int)
    parser.add_argument(
        "--vocab-size", default=40000, type=int, required=False)
    parser.add_argument(
        "--vocab-min-count", default=1, type=int, required=False)
    parser.add_argument(
        "--vocab-lower", default=1, type=int, choices=[0,1], required=False)
    parser.add_argument(
        "--pad-input", default=1, type=int, choices=[0,1], required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None, required=False, type=str)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.pad_input:
        print("hi!")
        print(max(args.filter_widths))
        pad_size = int(max(args.filter_widths) / 2)
        left_pad = ["_LPAD_"] * pad_size
        right_pad = ["_RPAD_"] * pad_size
    else:
        left_pad = None
        right_pad = None


    from dataio.recipes.classification import read_sequence_label_data
    from reader import LabelReader, DiscreteSequenceReader
    
    input_reader = DiscreteSequenceReader(field=args.input_field,
        top_k=args.vocab_size, at_least=args.vocab_min_count,
        lowercase=args.vocab_lower, left_pad=left_pad, right_pad=right_pad)
    label_reader = LabelReader(field=args.target_field)
    
    data_train = read_sequence_label_data(
        args.train, input_reader, label_reader, skip_header=True, 
        collect_stats=True, batch_size=args.batch_size, gpu=args.gpu)

    data_valid = read_sequence_label_data(
        args.valid, input_reader, label_reader, skip_header=True, 
        collect_stats=False, batch_size=args.batch_size, gpu=args.gpu)


    from input_module import DiscreteFeatureSequenceInput
    from encoder import CNNEncoder
    input_module = DiscreteFeatureSequenceInput(
        input_reader.vocab.size, args.embedding_size)
    encoder = CNNEncoder(
        input_module, args.filter_widths, args.filter_size,
        dropout=args.dropout, layer_norm=args.layer_norm)
    from mlp import MLP
    mlp = MLP(args.filter_size * len(args.filter_widths),
              label_reader.vocab.size,)
              #[100,20], ["relu", "relu"])

    from models.classification_model_base import ClassificationModel 
    
    model = ClassificationModel(encoder, mlp)


    #model.set_meta("model_type", "seq2seq_rnn")
    #model.set_meta("encoder_vocab", vocab_src)
    #model.set_meta("decoder_vocab", vocab_tgt)

    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)
        model = model.cuda(args.gpu)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, 
            weight_decay=args.l2)
    elif args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr,
            weight_decay=args.l2)
    elif args.optimizer == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr,
            weight_decay=args.l2)
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
            weight_decay=args.l2)
    else:
        raise Exception("Unkown optimizer: {}".format(args.optimizer))

    if args.save_model is not None:
        d = os.path.dirname(args.save_model)
        if d != "" and not os.path.exists(d):
            os.makedirs(d)

    weight=None
    #weight = torch.FloatTensor(
    #    [1., 2.9964664310954063, 3.129151291512915]).cuda(1)

    crit = criterion.CrossEntropy(model, optimizer, weight=weight)
    crit = criterion.PrecRecallReporter(crit, label_reader.vocab.index2token_)
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
