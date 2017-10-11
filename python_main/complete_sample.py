import argparse
from dataio.recipes import read_sequence_data
import torch
import random
import textwrap

def pretty_print_th(vocab, lt):
    return [vocab.token(idx) for idx in lt.data if idx > 0]


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
    input_reader = model.get_meta("input_reader")
    vocab = input_reader.vocab

    data = read_sequence_data(
        args.data, input_reader, skip_header=args.skip_header,
        collect_stats=False, batch_size=args.batch_size, gpu=args.gpu)

    for batch in data.iter_batch():
        for prefix in args.prefix:
            decoder_inputs = [di[:,:prefix + 1] for di in batch.decoder_inputs]
            batch_sequences = model.complete_sequence(
                decoder_inputs, max_steps=args.max_steps,
                beam_size=args.beam_size)
        for b in range(batch.decoder_inputs[0].size(0)):
            print("")
            tokens = pretty_print_th(vocab, batch.decoder_inputs[0][b])
            print("complete: ", " ".join(tokens))
            print("input: ", " ".join(tokens[:prefix + 1]))
            for i, (score, seq) in enumerate(batch_sequences[b], 1):
                pred = " ".join([vocab.token(idx) for idx in seq])
                print("predicted ({}. {:0.3f}): {}".format(i, score, pred)) 

            input()
        
            
     


