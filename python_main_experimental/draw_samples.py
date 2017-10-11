import argparse
import dataio
import torch
import random
import textwrap


def predict(model_type, model, batch, max_steps, set_rs=None):
    if model_type == "seq2seq_rnn":
        return model.greedy_predict(
            batch.encoder_inputs, batch.encoder_input_length, 
            max_steps=max_steps)
    elif model_type == "seq2seq_rnn_rs":
        rs = batch.decoder_inputs[1].data[:,0]
        if set_rs is not None:
            rs.fill_(set_rs)

        return model.greedy_predict(
            batch.encoder_inputs, batch.encoder_input_length, 
            max_steps=max_steps,
            extra_inputs=rs)
    elif model_type == "seq2seq_rnn_ts":
        ts = batch.decoder_inputs[1].data[:,0]
        if set_rs is not None:
            ts.fill_(set_rs)

        return model.greedy_predict(
            batch.encoder_inputs, batch.encoder_input_length, 
            max_steps=max_steps,
            extra_inputs=ts)

    else:
        raise Exception("Bad model type: {}".format(model_type))




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Eval seq2seq.")

    parser.add_argument("--data", required=True)

    parser.add_argument("--model", required=True)
    parser.add_argument("--max-steps", required=False, default=100, type=int)

    parser.add_argument("--gpu", default=-1, type=int, required=False)
    parser.add_argument("--batch-size", required=False, type=int, default=4)
    parser.add_argument("--set-rs", default=None, required=False, type=int)

    args = parser.parse_args()

#    random.seed(args.seed)
#    torch.manual_seed(args.seed)
    model = torch.load(args.model, map_location=lambda storage, loc: storage)

    vocab_src = model.get_meta("encoder_vocab")
    vocab_tgt = model.get_meta("decoder_vocab")

    model_type = model.get_meta("model_type")

    if model_type == "seq2seq_rnn":
        data = dataio.read_seq2seq_dataset(
            args.data, 0, vocab_src, 1, vocab_tgt,
            skip_header=True, batch_size=args.batch_size, gpu=args.gpu)
    elif model_type == "seq2seq_rnn_rs":
        data = dataio.read_seq2seq_rs_dataset(
            args.data, 0, vocab_src, 1, vocab_tgt,
            max_steps=model.get_meta("max_steps"), skip_header=True,
            batch_size=args.batch_size, gpu=args.gpu)
    elif model_type == "seq2seq_rnn_ts":
        data = dataio.read_seq2seq_ts_dataset(
            args.data, 0, vocab_src, 1, vocab_tgt,
            max_steps=model.get_meta("max_steps"), skip_header=True,
            batch_size=args.batch_size, gpu=args.gpu)


    if args.gpu > -1:
        #with torch.cuda.device(args.gpu):
            #torch.cuda.manual_seed(args.seed)
        model.cuda(args.gpu)

    model.eval()

    import random
    random.seed(34535)

    for batch in data.iter_batch():
        inputs = vocab_src.inv_lookup_matrix(batch.encoder_inputs[0].data)
        targets = vocab_tgt.inv_lookup_matrix(batch.target.data)
#        predictions = predict(
#            model_type, model, batch, max_steps=args.max_steps,
#            set_rs=args.set_rs)
#        predicted_targets = vocab_tgt.inv_lookup_matrix(predictions)

        beam_size = 64
        if model.get_meta("model_type") == "seq2seq_rnn":
            batch_beam_predictions = model.beam_search(
                batch.encoder_inputs, batch.encoder_input_length, beam_size=beam_size)
        else:
            batch_size = batch.target.size(0)
            rs = batch.decoder_inputs[1].data.new(batch_size * beam_size)

            for b in range(batch_size):
                #rs[b * beam_size: (b+1) * beam_size].fill_(batch.decoder_inputs[1].data[b,0])
                rs[b * beam_size: (b+1) * beam_size].fill_(args.set_rs)

            batch_beam_predictions = model.beam_search(
                batch.encoder_inputs, batch.encoder_input_length, extra_inputs=rs, beam_size=beam_size)



        print("")
        for i, input_tokens in enumerate(inputs):
            input_string = "           input: {}".format(
                " ".join(input_tokens))

            print(textwrap.fill(input_string, subsequent_indent=" " * 18))
            expected_string = " expected output: {}".format(
                " ".join(targets[i]))
            print(textwrap.fill(expected_string, subsequent_indent=" " * 18))
            #predicted_string = "predicted output: {}".format(
            #    " ".join(predicted_targets[i]))
            #print(textwrap.fill(predicted_string, subsequent_indent=" " * 18))
            #print("")
            beam_predictions = batch_beam_predictions[i]

            
            for score, beam_prediction in beam_predictions[:10]:
                if score == float("-inf"): continue
                print(score)
                print(beam_prediction)




            #attention = predictions.attention[i]
            #attention_vals, attention_idxs = torch.sort(attention, 1, True)
            x = input()
           
#            if x == "p":
#                for j, token in enumerate(predicted_targets[i]):
#                    print(predicted_targets[i][:j+1])
#                    for k in range(3):
#                        if attention_vals[j,k] == 0:
#                            break
#                        weight = attention_vals[j,k]
#                        index = attention_idxs[j,k]
#
#                        print(weight, index, input_tokens[index])
#                    print("")
#
#                input() 

