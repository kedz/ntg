local RNNDecoder = torch.class('ntg.module.RNNDecoder')

function RNNDecoder:__init(input_embedding, rnn, attention, predictor)
   self.input_embedding = input_embedding
   self.rnn = rnn
   self.attention = attention
   self.predictor = predictor
   self._networks = {}
   table.insert(self._networks, self.input_embedding)
   table.insert(self._networks, self.rnn)
   table.insert(self._networks, self.attention)
   table.insert(self._networks, self.predictor)

   if torch.type(rnn) == 'ntg.network.LSTM' 
         or torch.type(rnn) == 'ntg.network.StackedLSTM' then
      self.setState = function (self, state) 
         return {self.rnn:setInitOutputState(state[1]), 
                 self.rnn:setInitCellState(state[2])}
      end
      self.getGradState = function (self)
         return {self.rnn:getGradInitOutputState(), 
                 self.rnn:getGradInitCellState()}
      end
   else
      print("Bad rnn unit!")
      
      os.exit()
   end
end

function RNNDecoder:description(offset)
   offset = offset or ''
   local buffer = offset .. 'Module: RNNDecoder\n' 
      .. offset .. self.tokenDecoder:description(offset .. '| ') 
      .. '\n' 
      .. offset .. '      V V          V V          V V\n'
      .. offset .. '      V V          V V          V V\n'
      .. offset .. '      V V          V V          V V\n'
      .. offset .. self.rnn:description(offset .. '| ')
   return buffer
end

function RNNDecoder:parameters()
   local params = {}
   local grad_params = {}
   for i=1,#self._networks do
      local net_params, net_grad_params = self._networks[i]:parameters()
      ntg.util.Table.append(params, net_params)
      ntg.util.Table.append(grad_params, net_grad_params)
   end
   return params, grad_params
end

function RNNDecoder:zeroGradParameters()
   for i=1,#self._networks do
      self._networks[i]:zeroGradParameters()
   end
end

function RNNDecoder:forward(state, context, batch)

   self:setState(state)
   
   self.input_embedded = self.input_embedding:forward(
      batch.target_input, batch.target_mask)
   
   self.rnn_output = self.rnn:forward(self.input_embedded, batch.target_mask)

   self.predictor_input = self.attention:forward(context, self.rnn_output,
      batch.source_mask, batch.target_mask)

   self.output = self.predictor:forward(self.predictor_input, 
      batch.target_mask)
   
   return self.output

end

function RNNDecoder:backward(state, context, batch, grad_output)

   self.grad_att = self.predictor:backward(
      self.predictor_input, grad_output, batch.target_mask)

   self.grad_context, self.grad_rnn_output = self.attention:backward(
      context, self.rnn_output, 
      self.grad_att, batch.source_mask, batch.target_mask)

   self.grad_input_embedded = self.rnn:backward(
      self.input_embedded, self.grad_rnn_output, batch.target_mask)
   self.grad_state = self:getGradState()         

   self.input_embedding:backward(
      batch.target_input, self.grad_input_embedded, batch.target_mask)

   return self.grad_state, self.grad_context

end


