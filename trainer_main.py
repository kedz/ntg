import sys
import argparse
from dataio import extract_vocab_from_tsv, extract_dataset_from_tsv, load_data
import model_builder
import trainer

import random
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train seq2seq.")

    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument(
        "--rnn", required=False, type=str, default="rnn",
        choices=["rnn", "gru", "lstm"])
    parser.add_argument("--num-layers", default=1, type=int, required=False)
    parser.add_argument("--bidirectional", default=1, type=int, 
        choices=[0,1])
    parser.add_argument("--lr", required=False, default=.001, type=float)
    parser.add_argument("--batch-size", default=16, type=int, required=False)
    parser.add_argument(
        "--embedding-dim", required=False, type=int, default=300)
    parser.add_argument(
        "--hidden-dim", required=False, type=int, default=300)
    parser.add_argument(
        "--attention", required=False, choices=["none", "dot",],
        default="dot")
    parser.add_argument(
        "--src-tgt-fields", required=False, default=(0, 1,), type=int,
        nargs=2)

    parser.add_argument(
        "--length-mode", default="none", type=str,
        choices=["none", "decoder-input-embedding1"])
    parser.add_argument("--length-field", required=False, default=2, type=int)
    parser.add_argument(
        "--len-embedding-dim", default=50, required=False, type=int)

    parser.add_argument("--gpu", default=-1, type=int, required=False)
    parser.add_argument("--epochs", default=25, type=int, required=False)
    parser.add_argument("--seed", default=83419234, type=int, required=False)
    parser.add_argument(
        "--optimizer", default="adagrad", type=str, required=False,
        choices=["sgd", "adagrad", "adadelta", "adam"])
    parser.add_argument("--dropout", default=0.0, required=False, type=float)
    parser.add_argument("--save-path", default=None, required=False, type=str)
    parser.add_argument(
        "--enc-vocab-size", default=40000, type=int, required=False)
    parser.add_argument(
        "--dec-vocab-size", default=40000, type=int, required=False)

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpu > -1:
        with torch.cuda.device(args.gpu):
            torch.cuda.manual_seed(args.seed)

    src_field, tgt_field = args.src_tgt_fields
    vocab, datasets = load_data(
        args.train, args.valid, src_field=src_field, tgt_field=tgt_field,
        length_mode=args.length_mode, len_field=args.length_field,
        enc_vocab_size=args.enc_vocab_size, dec_vocab_size=args.dec_vocab_size)

    data_train, data_valid = datasets
    src_vocab = vocab[0]
    tgt_vocab = vocab[1]


    n_tokens_enc_train = data_train.source.gt(0).sum()
    n_unk_enc_train = data_train.source.eq(src_vocab.unknown_index).sum()
    per_unk_enc_train = n_unk_enc_train / n_tokens_enc_train
    
    n_tokens_dec_train = data_train.target_in.gt(0).sum()
    n_unk_dec_train = data_train.target_in.eq(tgt_vocab.unknown_index).sum()
    per_unk_dec_train = n_unk_dec_train / n_tokens_dec_train

    n_tokens_enc_valid = data_valid.source.gt(0).sum()
    n_unk_enc_valid = data_valid.source.eq(src_vocab.unknown_index).sum()
    per_unk_enc_valid = n_unk_enc_valid / n_tokens_enc_valid
    
    n_tokens_dec_valid = data_valid.target_in.gt(0).sum()
    n_unk_dec_valid = data_valid.target_in.eq(tgt_vocab.unknown_index).sum()
    per_unk_dec_valid = n_unk_dec_valid / n_tokens_dec_valid

    print("\nTraining data")
    print("=============")
    print("# encoder tokens: {}".format(n_tokens_enc_train))
    print("% encoder unk tokens: {}".format(per_unk_enc_train))
    print("# decoder tokens: {}".format(n_tokens_dec_train))
    print("% decoder unk tokens: {}".format(per_unk_dec_train))

    print("\nValidation data")
    print("===============")
    print("# encoder tokens: {}".format(n_tokens_enc_valid))
    print("% encoder unk tokens: {}".format(per_unk_enc_valid))
    print("# decoder tokens: {}".format(n_tokens_dec_valid))
    print("% decoder unk tokens: {}".format(per_unk_dec_valid))

    print("\nbuilding model...")

    print(args.bidirectional == 1)
    if len(vocab) == 2:
        vocab_src, vocab_tgt = vocab
        model = model_builder.seq2seq(
            vocab_src, vocab_tgt, 
            rnn_type=args.rnn, num_layers=args.num_layers,
            bidirectional=args.bidirectional == 1,
            embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
            attention_type=args.attention, dropout=args.dropout)
    else:
        vocab_src, vocab_tgt, vocab_len = vocab
        model = model_builder.seq2seq_lt(
            vocab_src, vocab_tgt, vocab_len,
            rnn_type=args.rnn, num_layers=args.num_layers,
            bidirectional=args.bidirectional == 1,
            embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
            attention_type=args.attention, dropout=args.dropout,
            len_embedding_dim=args.len_embedding_dim)



#    print("reading training data...")
#    data_train = extract_dataset_from_tsv(args.train, vocab_src, vocab_tgt)
    data_train.set_batch_size(args.batch_size)
    data_train.set_gpu(args.gpu)
#
#    print("reading validation data...")
#    data_valid = extract_dataset_from_tsv(args.valid, vocab_src, vocab_tgt)
    data_valid.set_batch_size(args.batch_size)
    data_valid.set_gpu(args.gpu)
#
#    print("building model...")
#    model = model_builder.seq2seq(
#        vocab_src, vocab_tgt, rnn_type=args.rnn, 
#        embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
#        attention_type=args.attention, dropout=args.dropout)

    if args.gpu > -1:
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

    trainer.train(model, data_train, data_valid, optimizer, args.epochs,
        best_model_path=args.save_path)
    
    exit()

    
    for batch in data_train.iter_batch():
        pred_tf = model.predict(batch, teacher_forcing=True)
        pred_gr = model.predict(batch, teacher_forcing=False, beam_size=1)
        input_str = vocab_src.inv_lookup_matrix(batch.source.data)
        output_str = vocab_tgt.inv_lookup_matrix(batch.target_out.data)
        pred_tf_str = vocab_tgt.inv_lookup_matrix(pred_tf)
        pred_gr_str = vocab_tgt.inv_lookup_matrix(pred_gr)
        for input, gold, pred_tf, pred_gr in zip(
                input_str, output_str, pred_tf_str, pred_gr_str):

            print("input\n=====")
            print("    {}".format(" ".join(input)))
            print("")
            print("output (correct)")
            print("================")
            print("    {}".format(" ".join(gold)))
            print("")
            print("output (teacher forcing)")
            print("========================")
            print("    {}".format(" ".join(pred_tf)))
            print("")
            print("output (greedy)")
            print("===============")
            print("    {}".format(" ".join(pred_gr)))
            print("")
            print("")
    exit()


