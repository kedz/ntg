import os
import argparse
import json

from dataio.reader import DiscreteSequenceReader
from dataio.recipes import read_sequence_data
import input_module
import models
import criterion
import trainer

import random
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an lvrnn seq2seq model.")

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
        "--rnn-hidden-size", required=False, type=int, default=300)
    parser.add_argument(
        "--bridge-type", required=False, type=str, default="linear-relu",
        choices=["average", "linear-relu"])
    parser.add_argument(
        "--attention-type", required=False, choices=["none", "dot",],
        default="dot")
    parser.add_argument(
        "--learn-init", default=1, required=False, type=int, choices=[0,1])

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
        "--input-dropout", default=0.0, required=False, type=float)

    parser.add_argument(
        "--source-field", required=False, default=0, type=int)
    parser.add_argument(
        "--target-field", required=False, default=1, type=int)
    parser.add_argument(
        "--source-vocab-size", default=80000, type=int, required=False)
    parser.add_argument(
        "--target-vocab-size", default=40000, type=int, required=False)
    parser.add_argument(
        "--enc-start-token", type=str, required=False, default="_ESTART_")
    parser.add_argument(
        "--dec-start-token", type=str, required=False, default="_DSTART_")
    parser.add_argument(
        "--dec-stop-token", type=str, required=False, default="_DSTOP_")
    parser.add_argument(
        "--unknown-token", type=str, required=False, default="_UNK_")
    parser.add_argument(
        "--target-vocab-lower", type=int, choices=[0,1], default=1, 
        required=False)
    parser.add_argument(
        "--target-vocab-min-count", default=1, type=int, required=False)
    parser.add_argument(
        "--source-vocab-lower", type=int, choices=[0,1], default=1, 
        required=False)
    parser.add_argument(
        "--source-vocab-min-count", default=1, type=int, required=False)

    parser.add_argument(
        "--skip-header", default=1, type=int, choices=[0,1], required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None, required=False, type=str)

    parser.add_argument(
        "--feature-fields", nargs="+", type=int, required=False, default=())
    parser.add_argument(
        "--feature-embedding-size", type=int, required=False, default=100)
    parser.add_argument(
        "--missing-feature-value", type=str, required=False, default="MISSING")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set random seed state
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)

    # Create field reader to load source input sequences from a tsv field.
    # This reader also does minimal preprocessing by default, i.e. lowercasing
    # tokens, replacing digits with D, and replacing rare words with the 
    # unknown token.
    source_reader = DiscreteSequenceReader(
        field=args.source_field,
        top_k=args.source_vocab_size, at_least=args.source_vocab_min_count,
        lowercase=args.source_vocab_lower, left_pad=args.enc_start_token, 
        right_pad=None,
        unknown_token=args.unknown_token,
        offset_output=False)

    # Create field reader to load target input sequences from a tsv field.
    target_reader = DiscreteSequenceReader(
        field=args.target_field,
        top_k=args.target_vocab_size, at_least=args.target_vocab_min_count,
        lowercase=args.target_vocab_lower, left_pad=args.dec_start_token, 
        right_pad=args.dec_stop_token,
        unknown_token=args.unknown_token,
        offset_output=True)

    feature_readers = [DiscreteFeatureReader(
                           field=field, 
                           missing_token=args.missing_feature_value)
                       for field in args.feature_fields]

    # Read training and validation data from tsv files.
    # collect_stats=True for the training dataset so that vocab statistics
    # are collected and rare words can be replaced with unknown token.
    data_train = read_sequence_data(
        args.train, target_reader, 
        input_sequence_readers=source_reader,
        feature_readers=feature_readers,
        skip_header=args.skip_header,
        collect_stats=True, batch_size=args.batch_size, gpu=args.gpu)

    data_valid = read_sequence_data(
        args.valid, target_reader, 
        input_sequence_readers=source_reader,
        feature_readers=feature_readers,
        skip_header=args.skip_header,
        collect_stats=False, batch_size=args.batch_size, gpu=args.gpu)

    # Create the encoder/decoder input control logic. 
    # input_module classes can work
    # on an entire sequence (i.e. training) and work a timestep at a time
    # during prediction (using either greedy or beam search).
    encoder_input = input_module.InputGroup()
    encoder_input.add_discrete_sequence(
        source_reader.vocab.size, 
        args.embedding_size, 
        dropout=args.input_dropout)

    decoder_input = input_module.InputGroup()
    decoder_input.add_discrete_sequence(
        target_reader.vocab.size, 
        args.embedding_size, 
        dropout=args.input_dropout)
    for feature_reader in feature_readers:
        decoder_input.add_discrete_feature(
            feature_reader.vocab.size,
            args.feature_embedding_size,
            dropout=args.dropout)

    model = models.RNNSeq2Seq.from_args(
        args, 
        encoder_input, 
        decoder_input,
        target_vocab_size=target_reader.vocab.size)
 
    model.set_meta("model_type", "rnnseq2seq")
    model.set_meta("source_reader", source_reader)
    model.set_meta("target_reader", target_reader)

    if args.gpu > -1:
        model = model.cuda(args.gpu)

    # Set up optimization problem.
    optimizer = trainer.optimizer_from_args(args, model.parameters())
    crit = criterion.SequenceCrossEntropy(model, optimizer)

    # Run optimization on dataset.
    results = trainer.minimize_criterion(
        crit, data_train, data_valid, args.epochs,
        save_best_model=args.save_model)

    best_valid_loss = min(results["valid_nll"])
    best_epoch = results["valid_nll"].index(best_valid_loss) + 1

    print("\nBest epoch: {}".format(best_epoch))
    print("Best validation nll: {}".format(best_valid_loss))

    if args.save_results is not None:
        print("Writing results json to {} ...".format(args.save_results))
        results["parameters"] = vars(args)
        with open(args.save_results, "w") as fp:
            json.dump(results, fp)
