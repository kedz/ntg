import sys
import argparse
from dataio import extract_vocab_from_tsv, extract_dataset_from_tsv
from encoder import RNNEncoder
from decoder import RNNDecoder
from base_model import Seq2SeqModel

import math
import random
random.seed(1986)
import torch
torch.manual_seed(1986)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train CNN classifier.")

    parser.add_argument("--train", required=True)
    parser.add_argument("--valid", required=True)
    parser.add_argument("--lr", required=False, default=.001, type=float)
    parser.add_argument("--batch-size", default=16, type=int, required=False)
    parser.add_argument(
        "--embedding-dim", required=False, type=int, default=300)
    parser.add_argument(
        "--attention", required=False, choices=["none", "dot",],
        default="dot")
    parser.add_argument(
        "--src-tgt-fields", required=False, default=(0, 1,), type=int,
        nargs=2)
    parser.add_argument("--meta-field", required=False, default=None, type=int)

    args = parser.parse_args()

    vocab_src, vocab_tgt = extract_vocab_from_tsv(
        args.train, 
        source_field=args.src_tgt_fields[0],
        target_field=args.src_tgt_fields[1])
    data_train = extract_dataset_from_tsv(args.train, vocab_src, vocab_tgt)
    data_train.set_batch_size(args.batch_size)

    enc = RNNEncoder(vocab_src.size, args.embedding_dim)
    dec = RNNDecoder(vocab_tgt, args.embedding_dim, 
        attention_mode=args.attention)
    

    model = Seq2SeqModel(enc, dec)

    model.train()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr)
    import torch.nn.functional as F


    for e in range(15):
        total_loss = 0
        total_ex = 0
        max_batches = math.ceil(data_train.size / args.batch_size) 
        for i, batch in enumerate(data_train.iter_batch(), 1):
            optimizer.zero_grad()

            logits = model(batch)
            logits_flat = logits.view(
                logits.size(0) * logits.size(1), logits.size(2))
            
            tgt_out = batch.target_out
            tgt_out_flat = tgt_out.t().contiguous().view(
                tgt_out.size(0) * tgt_out.size(1))        
            
            loss = F.cross_entropy(logits_flat, tgt_out_flat, ignore_index=0)
            total_loss += loss.data[0] * batch.source.data.size(0)
            total_ex += batch.source.data.size(0)
            sys.stdout.write("\r{}/{} ({:0.3f}%) nll={:0.3f}".format(
                i, max_batches, i / max_batches * 100, 
                total_loss / total_ex))
            sys.stdout.flush()


            loss.backward()            
            optimizer.step()
        print("")
        print(e, total_loss / total_ex)        


    
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


    for i, batch in enumerate(dataset.batch_iter(batch_size), 1):
        status_update("train", 100 * i / num_batches, avg_nll, acc)
       
        inputs, targets = batch
        logits = model(inputs, is_train=True)
        loss = F.cross_entropy(logits, targets) 
        loss.backward()
        optimizer.step()
    
        #if model.steps == report_step:
        #    model.print_activation_info()
        #    report_step *= 2



        total_inputs += inputs.data.size(0)
        pred_targets = logits.data.max(1)[1]
        total_correct += (pred_targets == targets.data).sum()
        total_nll += loss.data[0] * inputs.data.size(0)
        avg_nll = total_nll / total_inputs
        acc = 100 * total_correct / total_inputs
    status_update("train", 100, avg_nll, acc)

