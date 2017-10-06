import argparse
import os
import json

import dataio
from dataio.recipes import read_sequence_data
import input_module
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
        "--rnn-hidden-size", required=False, type=int, default=300)

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
        "--learn-init", default=1, required=False, type=int, choices=[0,1])

    parser.add_argument(
        "--input-field", required=False, default=0, type=int)
    parser.add_argument(
        "--vocab-size", default=40000, type=int, required=False)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None,required=False, type=str)
    parser.add_argument(
        "--start-pad", type=str, required=False, default="_START_")
    parser.add_argument(
        "--stop-pad", type=str, required=False, default="_STOP_")
    parser.add_argument(
        "--unknown-token", type=str, required=False, default="_UNK_")
    parser.add_argument(
        "--vocab-lower", type=int, choices=[0,1], default=1, required=False)
    parser.add_argument(
        "--vocab-min-count", default=1, type=int, required=False)
    parser.add_argument(
        "--skip-header", default=1, type=int, choices=[0,1], required=False)

    args = parser.parse_args()

    # Set random seed state
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)

    # Create field reader to load input sequences from a tsv field.
    # This reader also does minimal preprocessing by default, i.e. lowercasing
    # tokens, replacing digits with D, and replacing rare words with the 
    # unknown token.
    input_reader = dataio.reader.DiscreteSequenceReader(
        field=args.input_field,
        top_k=args.vocab_size, at_least=args.vocab_min_count,
        lowercase=args.vocab_lower, left_pad=args.start_pad, 
        right_pad=args.stop_pad,
        unknown_token=args.unknown_token,
        offset_output=True)

    # Read training and validation data from tsv files.
    # collect_stats=True for the training dataset so that vocab statistics
    # are collected and rare words can be replaced with unknown token.
    data_train = read_sequence_data(
        args.train, input_reader, skip_header=args.skip_header,
        collect_stats=True, batch_size=args.batch_size, gpu=args.gpu)

    data_valid = read_sequence_data(
        args.valid, input_reader, skip_header=args.skip_header,
        collect_stats=False, batch_size=args.batch_size, gpu=args.gpu)

    # Create the decoder input control logic. input_module classes can work
    # on an entire sequence (i.e. training) and work a timestep at a time
    # during prediction (using either greedy or beam search).
    decoder_input = input_module.InputGroup()
    decoder_input.add_discrete_sequence(
        input_reader.vocab.size, 
        args.embedding_size, 
        dropout=args.input_dropout)

    # Create the language model.
    model = models.RNNLM.from_args(
        args, 
        decoder_input, 
        target_vocab_size=input_reader.vocab.size)

    model.set_meta("model_type", "rnnlm")
    model.set_meta("input_reader", input_reader)

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
