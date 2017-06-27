local RNNEncoder = torch.class('ntg.module.RNNEncoder')

function RNNEncoder:__init(tokenEncoder, rnn)
   self.tokenEncoder = tokenEncoder
   self.rnn = rnn
   
   if torch.type(rnn) == 'ntg.network.LSTM' then
      self.getState = function (self, batch) 
         return {self.rnn:getOutputState(), self.rnn:getCellState()}
      end
      self.setGradState = function (self, batch, gradState)
         self.rnn:setOutputGradState(gradState[1])
         self.rnn:setCellGradState(gradState[2])
      end
   elseif torch.type(rnn) == 'ntg.network.StackedLSTM' then
      self.getState = function (self, batch) 
         return {self.rnn:getOutputState(), self.rnn:getCellState()}
      end

      self.setGradState = function (self, batch, gradState)
         self.rnn:setOutputGradState(gradState[1])
         self.rnn:setCellGradState(gradState[2])
      end
   elseif torch.type(rnn) == 'ntg.network.StackedBiLSTM' then
      self.getState = function (self, batch)
         return {self.rnn:getOutputState(nil, batch.source_start),
                 self.rnn:getCellState(nil, batch.source_start)}
      end
      self.setGradState = function (self, batch, gradState)
         self.rnn:setOutputGradState(gradState[1], nil, batch.source_start)
         self.rnn:setCellGradState(gradState[2], nil, batch.source_start)
      end
   else
       print("AHHHHH")
       os.exit()
   end

end

function RNNEncoder:description(offset)
   offset = offset or ''
   local buffer = offset .. 'Module: RNNEncoder\n' 
      .. offset .. self.tokenEncoder:description(offset .. '| ') 
      .. '\n' 
      .. offset .. '      V V          V V          V V\n'
      .. offset .. '      V V          V V          V V\n'
      .. offset .. '      V V          V V          V V\n'
      .. offset .. self.rnn:description(offset .. '| ')
   return buffer
end

function RNNEncoder:parameters()
   local params = {}
   local gradParams = {}
   local tokenEncParams, tokenEncGradParams = self.tokenEncoder:parameters()
   local rnnParams, rnnGradParams = self.rnn:parameters()
   ntg.util.Table.append(params, tokenEncParams)
   ntg.util.Table.append(gradParams, tokenEncGradParams)
   ntg.util.Table.append(params, rnnParams)
   ntg.util.Table.append(gradParams, rnnGradParams)
   return params, gradParams
end


function RNNEncoder:forward(batch)

   self.input_embedded = self.tokenEncoder:forward(
      batch.source_input, batch.source_mask)

   self.context = self.rnn:forward(self.input_embedded, batch.source_mask)
   self.state = self:getState(batch)
   
   return self.state, self.context

end

function RNNEncoder:backward(batch, grad_state, grad_context)

   self:setGradState(batch, grad_state)

   self.grad_rnn = self.rnn:backward(
      self.input_embedded, grad_context, batch.source_mask)

   self.tokenEncoder:backward(
      batch.source_input, self.grad_rnn, batch.source_mask)

end

function RNNEncoder:zeroGradParameters()
   self.tokenEncoder:zeroGradParameters()
   self.rnn:zeroGradParameters()
end

function RNNEncoder:reset(std)
   self.tokenEncoder:reset(std)
   self.rnn:reset(std)
end
