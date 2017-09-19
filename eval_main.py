import sys
import argparse
from dataio import extract_vocab_from_tsv, extract_dataset_from_tsv, load_data
import model_builder
import trainer

import random
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Eval seq2seq.")

    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--test", required=True)

    parser.add_argument("--model", required=True)

    parser.add_argument("--gpu", default=-1, type=int, required=False)
    parser.add_argument("--seed", default=83419234, type=int, required=False)
    parser.add_argument("--batch-size", required=False, type=int, default=4)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = torch.load(args.model)
    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)
            model.cuda(args.gpu)

    else:
        model.float()
            
    vocab_src, vocab_tgt = model.encoder.vocab_, model.decoder.vocab
    if hasattr(model.decoder, "vocab_len_"):
        vocab_len = model.decoder.vocab_len_
    else:
        vocab_len = None

    data_train = extract_dataset_from_tsv(
        args.train, vocab_src, vocab_tgt, vocab_len=vocab_len)
    data_valid = extract_dataset_from_tsv(
        args.valid, vocab_src, vocab_tgt, vocab_len=vocab_len)
    data_test = extract_dataset_from_tsv(
        args.test, vocab_src, vocab_tgt, vocab_len=vocab_len)

    data_train.set_batch_size(args.batch_size)
    data_train.set_gpu(args.gpu)
    data_valid.set_batch_size(args.batch_size)
    data_valid.set_gpu(args.gpu)
    data_test.set_batch_size(args.batch_size)
    data_test.set_gpu(args.gpu)
 
    trainer.eval_epoch(model, data_train) 
    trainer.eval_epoch(model, data_valid) 
    trainer.eval_epoch(model, data_test) 
    

