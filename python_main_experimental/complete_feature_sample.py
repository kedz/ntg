import argparse
from dataio.recipes import read_sequence_data
import torch
import random
import textwrap
from torch.autograd import Variable
from itertools import product

def pretty_print_th(vocab, lt):
    return [vocab.token(idx) for idx in lt.data if idx > 0]


def vector2tokens(vocab, vec):
    if isinstance(vec, Variable):
        vec = vec.data
    assert vec.dim() == 1
    return [vocab.token(idx) for idx in vec if idx > 0]

def vector2string(vocab, vec):
    return " ".join(vector2tokens(vocab, vec))



def complete_lm(model, batch, prefixes):
    pass


def complete_seq2seq(model, batch, prefixes):
    pass

def make_prefix_data(data, prefix):
    return [var[:,:prefix + 1] if var.dim() == 2 else var
            for var in data]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Eval seq2seq.")

    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", required=False, default=100, type=int)
    parser.add_argument("--gpu", default=-1, type=int, required=False)
    parser.add_argument("--batch-size", required=False, type=int, default=8)
    parser.add_argument("--beam-size", required=False, type=int, default=8)
    parser.add_argument(
        "--prefix", nargs="+", default=(10,), type=int, required=False)
    parser.add_argument(
        "--skip-header", default=1, required=False, type=int, choices=[0,1])

    args = parser.parse_args()

    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    model.eval()

    if model.get_meta("model_type") == "rnnlm":
        source_reader = None
        source_vocab = None
        target_reader = model.get_meta("input_reader")
        target_vocab = input_reader.vocab
        feature_readers = model.meta.get("feature_readers", None)
    else:
        source_reader = model.get_meta("source_reader")
        source_vocab = source_reader.vocab
        target_reader = model.get_meta("target_reader")
        target_vocab = target_reader.vocab
        feature_readers = model.meta.get("feature_readers", None)

    data = read_sequence_data(
        args.data, target_reader, skip_header=args.skip_header,
        input_sequence_readers=source_reader,
        feature_readers=feature_readers,
        collect_stats=False, batch_size=1, gpu=args.gpu)

    sweep_feature = 0
    #TODO fix vocab size to know about zero offset
    feature_size = feature_readers[sweep_feature].vocab.size - 1

    sweep_features = torch.LongTensor([x for x in range(1, feature_size + 1)])

    for batch in data.iter_batch():

        encoder_length = batch.encoder_length.repeat(feature_size)
        encoder_inputs = [batch.encoder_inputs[0].repeat(feature_size, 1)]
        decoder_inputs = [Variable(batch.decoder_inputs[0].data[:,:1].repeat(feature_size, 1))]
        decoder_features = []

        for f, df in enumerate(batch.decoder_features):
            if f == sweep_feature:
                decoder_features.append(Variable(sweep_features))
            else:
                decoder_features.append(df.repeat(feature_size))

        batch_size = batch.targets[0].size(0)
        prediction = model.complete_sequence(
            encoder_inputs,
            encoder_length,
            decoder_inputs,
            decoder_features,
            max_steps=args.max_steps,
            beam_size=args.beam_size)
        
        if len(batch.encoder_inputs) > 0:
            input_str = " ".join([source_vocab.token(idx) 
                                  for idx in batch.encoder_inputs[0].data[0] 
                                  if idx > 0])
            print("")
            print(textwrap.fill(
                "input:  {}".format(input_str), width=150, 
                subsequent_indent="       "))
            print("")


        for i in range(feature_size):
            for f, feature in enumerate(decoder_features):
                print("")
                print("feature {}: {}".format(
                    f, 
                    feature_readers[f].vocab.token(feature.data[i])))
            for score, pred_idx in prediction[i]:
                print(textwrap.fill(
                    "{:6.3f} | {}".format(
                        score, " ".join([target_vocab.token(idx) 
                                         for idx in pred_idx])),
                    subsequent_indent="          ", width=150))

        input()

        
#        exit()
#
#
#        for ex in range(batch_size):
#            if len(batch.encoder_inputs) > 0:
#                print("Encoder Input")
#                enc_input = vector2string(
#                    source_vocab, batch.encoder_inputs[0][ex])
#                print(textwrap.fill(enc_input))
#            print("Expected output:")
#            dec_output = vector2string(
#                target_vocab, batch.decoder_inputs[0][ex])
#            print(textwrap.fill(dec_output))
#
#            for prefix in args.prefix:
#                print("prefix ({})".format(prefix))
#                print(vector2string(
#                    target_vocab, prefix2input[prefix][0][ex]))
#                for score, pred in prefix2results[prefix][ex]:
#                    print("({:0.3f}) {}".format(
#                        score, " ".join([target_vocab.token(idx) 
#                                         for idx in pred])))
#
#            
#            
#            print("")
#            input()



#        for prefix in args.prefix:
#            decoder_inputs = [di[:,:prefix + 1] for di in batch.decoder_inputs]
#            batch_sequences = model.complete_sequence(
#                decoder_inputs, max_steps=args.max_steps,
#                beam_size=args.beam_size)
#        for b in range(batch.decoder_inputs[0].size(0)):
#            print("")
#            tokens = pretty_print_th(vocab, batch.decoder_inputs[0][b])
#            print("complete: ", " ".join(tokens))
#            print("input: ", " ".join(tokens[:prefix + 1]))
#            for i, (score, seq) in enumerate(batch_sequences[b], 1):
#                pred = " ".join([vocab.token(idx) for idx in seq])
#                print("predicted ({}. {:0.3f}): {}".format(i, score, pred)) 
#
#            input()
        
            
     


