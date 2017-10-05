from models.seq2seq_rnn import Seq2SeqRNN
from torch.autograd import Variable

class Seq2SeqRNNRS(Seq2SeqRNN):
    def __init__(self, *args, **keys):
        super(Seq2SeqRNNRS, self).__init__(*args, **keys)

    def setup_inputs(self, decoder_input_buffer, step, extra_inputs=None):
        rs = extra_inputs.new().resize_(extra_inputs.size(0), 1)
        rs[:,0].copy_(extra_inputs)
        rs.sub_(step)
        rs.masked_fill_(rs.lt(1), 1)
        decoder_input_step = decoder_input_buffer[:,step:step + 1]
        return [Variable(decoder_input_step), Variable(rs)]


