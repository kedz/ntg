import os
import argparse
import json

import dataio
from dataio.reader import LabelReader
from dataio.recipes import read_labeled_sequence_data
import input_module
import models
from mlp import MLP
import criterion
import trainer

import random
import torch

def validate_directory(path):
    path_dir = os.path.dirname(path)
    if path_dir != "" and not os.path.exists(path_dir):
        os.makedirs(path_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Train an rnn classifier from a pretrained seq2seq or lm model.")

    parser.add_argument(
        "--train", required=True)
    parser.add_argument(
        "--valid", required=True)

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
        "--input-field", required=False, default=0, type=int)
    parser.add_argument(
        "--target-field", required=False, default=1, type=int)

    parser.add_argument(
        "--skip-header", default=1, type=int, choices=[0,1], required=False)

    parser.add_argument(
        "--init-model", type=str, required=True)

    parser.add_argument(
        "--save-model", default=None, required=False, type=str)
    parser.add_argument(
        "--save-results", default=None, required=False, type=str)
    
    parser.add_argument(
        "--class-weights", default=None, nargs="+", type=str)

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

    init_model = torch.load(
        args.init_model, map_location=lambda storage, loc: storage)

    input_reader = init_model.get_meta("source_reader")
    input_reader.set_field(args.input_field)

    target_reader = LabelReader(field=args.target_field)
    dataio.tsv_reader.collect_tsv_stats(
        args.train, [target_reader], skip_header=args.skip_header,
        diagnostics=True)

    # Read training and validation data from tsv files.
    data_train = read_labeled_sequence_data(
        args.train, 
        input_reader, 
        target_reader,
        skip_header=args.skip_header,
        collect_stats=False, 
        batch_size=args.batch_size, 
        gpu=args.gpu)

    data_valid = read_labeled_sequence_data(
        args.valid, 
        input_reader, 
        target_reader,
        skip_header=args.skip_header,
        collect_stats=False, 
        batch_size=args.batch_size, 
        gpu=args.gpu)

    input_modules = init_model.encoder_input_modules
    encoder = init_model.encoder

    input_modules.set_dropout(args.input_dropout)
    encoder.set_dropout(args.dropout)

    mlp_input_size = 0
    for dim in encoder.rnn_state_dims:
        mlp_input_size += dim[0] * dim[2]
    mlp = MLP(mlp_input_size, target_reader.vocab.size, dropout=args.dropout)
    model = models.RNNClassifier(input_modules, encoder, mlp)
    model.init_state = init_model.init_state

    model.set_meta("model_type", "rnnclassifier")
    model.set_meta("input_reader", input_reader)
    model.set_meta("target_reader", target_reader)

    model.freeze_input_parameters()
    #model.freeze_encoder_parameters()


    if args.gpu > -1:
        model = model.cuda(args.gpu)

    if args.class_weights:

        labels = target_reader.labels
        weight = torch.FloatTensor(len(labels)).fill_(1)
        
        if args.gpu > -1:
            weight = weight.cuda(args.gpu)

        if len(args.class_weights) == 1 and args.class_weights[0] == "invfreq":
            for i, label in enumerate(labels):
                label_freq = target_reader.vocab.count[label]
                weight[i] = 1 / label_freq
        else:
            for class_weight in args.class_weights:
                label, label_weight = class_weight.split("=")
                label_weight = float(label_weight)
                weight[target_reader.vocab.index(label)] = label_weight

        print("label weights:")
        for label, label_weight in zip(labels, weight):
            print("{} = {:0.6f}".format(label, label_weight))

    else:
        weight = None

    # Set up optimization problem.
    optimizer = trainer.optimizer_from_args(args, model.thawed_parameters())
    crit = criterion.FMeasureCrossEntropy(
        model, optimizer, target_reader.vocab.size,
        weight=weight,
        class_names=target_reader.labels)

    # Run optimization on dataset.
    results = trainer.minimize_criterion(
        crit, data_train, data_valid, args.epochs,
        save_best_model=args.save_model, low_score=False)

    objective_values = [result["macro"]["f-measure"] 
                        for result in results["validation"]]
    best_valid_obj = max(objective_values)
    best_epoch = objective_values.index(best_valid_obj)

    print("\nBest epoch: {}\n".format(best_epoch + 1))
    
    
    print("Training nll: {}".format(
        results["training"][best_epoch]["nll"]))
    for label in target_reader.labels:
        print("{:15s} P={:6.3f} R={:6.3f} F={:6.3f}".format(
            label, 
            results["training"][best_epoch][label]["precision"],
            results["training"][best_epoch][label]["recall"],
            results["training"][best_epoch][label]["f-measure"]))
    print("{:25s} P={:6.3f} R={:6.3f} F={:6.3f}".format(
        "macro avg. unweighted", 
        results["training"][best_epoch]["macro"]["precision"],
        results["training"][best_epoch]["macro"]["recall"],
        results["training"][best_epoch]["macro"]["f-measure"]))

    print("")
    
    print("Validation nll: {}".format(
        results["validation"][best_epoch]["nll"]))
    for label in target_reader.labels:
        print("{:15s} P={:6.3f} R={:6.3f} F={:6.3f}".format(
            label, 
            results["validation"][best_epoch][label]["precision"],
            results["validation"][best_epoch][label]["recall"],
            results["validation"][best_epoch][label]["f-measure"]))
    print("{:25s} P={:6.3f} R={:6.3f} F={:6.3f}".format(
        "macro avg. unweighted", 
        results["validation"][best_epoch]["macro"]["precision"],
        results["validation"][best_epoch]["macro"]["recall"],
        results["validation"][best_epoch]["macro"]["f-measure"]))



    if args.save_results is not None:
        print("Writing results json to {} ...".format(args.save_results))
        results["parameters"] = vars(args)
        with open(args.save_results, "w") as fp:
            json.dump(results, fp)
