from models.seq2seq_rnn import Seq2SeqRNN
from torch.autograd import Variable

class Seq2SeqRNNTS(Seq2SeqRNN):
    def __init__(self, *args, **keys):
        super(Seq2SeqRNNTS, self).__init__(*args, **keys)

    def setup_inputs(self, decoder_input_buffer, step, extra_inputs=None):
        ts = extra_inputs.new().resize_(extra_inputs.size(0), 1)
        ts[:,0].copy_(extra_inputs)
        decoder_input_step = decoder_input_buffer[:,step:step + 1]
        return [Variable(decoder_input_step), Variable(ts)]


