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

def validate_directory(path):
    path_dir = os.path.dirname(path)
    if path_dir != "" and not os.path.exists(path_dir):
        os.makedirs(path_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an rnn seq2seq model.")

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
        "--l2-penalty", default=0.0, required=False, type=float)

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

    args = parser.parse_args()

    if args.save_model:
        validate_directory(args.save_model)
    if args.save_results:
        validate_directory(args.save_results)

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

    # Read training and validation data from tsv files.
    # collect_stats=True for the training dataset so that vocab statistics
    # are collected and rare words can be replaced with unknown token.
    data_train = read_sequence_data(
        args.train, target_reader, 
        input_sequence_readers=source_reader,
        skip_header=args.skip_header,
        collect_stats=True, batch_size=args.batch_size, gpu=args.gpu)

    data_valid = read_sequence_data(
        args.valid, target_reader, 
        input_sequence_readers=source_reader,
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

    objective_values = [result["nll"] for result in results["validation"]]
    best_valid_obj = min(objective_values)
    best_epoch = objective_values.index(best_valid_obj)

    print("\nBest epoch: {}\n".format(best_epoch + 1))
    print("Training nll: {}\n".format(
        results["training"][best_epoch]["nll"]))
    print("Validation nll: {}\n".format(
        results["validation"][best_epoch]["nll"]))

    if args.save_results is not None:
        print("Writing results json to {} ...".format(args.save_results))
        results["parameters"] = vars(args)
        with open(args.save_results, "w") as fp:
            json.dump(results, fp)
