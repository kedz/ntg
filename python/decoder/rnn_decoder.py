from decoder.decoder_base import DecoderBase

class RNNDecoder(DecoderBase):
    def __init__(self, input_module, rnn_module, attention_module, 
                 predictor_module, start_index, stop_index):
        super(RNNDecoder, self).__init__()

        self.input_module_ = input_module
        self.rnn_module_ = rnn_module
        self.attention_module_ = attention_module
        self.predictor_module_ = predictor_module
        self.start_index_ = start_index
        self.stop_index_ = stop_index

    @property
    def start_index(self):
        return self.start_index_

    @property
    def stop_index(self):
        return self.stop_index_

    @property
    def input_module(self):
        return self.input_module_

    @property
    def rnn_module(self):
        return self.rnn_module_

    @property
    def attention_module(self):
        return self.attention_module_

    @property
    def predictor_module(self):
        return self.predictor_module_

    def forward(self, inputs, init_state=None, context=None): 

        input_sequence = self.input_module(inputs)
        rnn_output, rnn_state = self.rnn_module(input_sequence, init_state)
        attention_output, _ = self.attention_module(
            rnn_output, context=context)

        max_steps = rnn_output.size(0)
        batch_size = rnn_output.size(1)

        attention_output_flat = attention_output.view(
            max_steps * batch_size, attention_output.size(2))

        logits_flat = self.predictor_module(attention_output_flat)
        logits = logits_flat.view(max_steps, batch_size, logits_flat.size(1))

        logits_mask = inputs[0].data.t().eq(0).view(
            max_steps, batch_size, 1).expand(
            max_steps, batch_size, logits.size(2))
        logits.data.masked_fill_(logits_mask, 0)

        return logits

    def greedy_predict(self, init_state, context):
        pass

    def forward_step(self, inputs, prev_state=None, context=None):

        input_sequence = self.input_module(inputs)
        rnn_output, rnn_state = self.rnn_module(input_sequence, prev_state)
        attention_output, weights = self.attention_module(
            rnn_output, context=context)
        logits = self.predictor_module(attention_output[0])
        return logits, weights, rnn_state



