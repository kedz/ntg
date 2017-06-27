local StackedLSTM, parent = torch.class('ntg.network.StackedLSTM', 
   'ntg.network.Network')

function StackedLSTM:__init(layerSize, inputSize, hiddenSize, revIn, revOut)

   self.layerSize = layerSize
   self.inputSize = inputSize
   self.hiddenSize = hiddenSize
   self.outputSize = hiddenSize
   self.reverseInput = revIn
   if self.reverseInput == nil then self.reverseInput = false end
   self.reverseOutput = revOut
   if self.reverseOutput == nil then self.reverseOutput = false end
 
   local outputSize = hiddenSize
   local L, D, H, R = layerSize, inputSize, hiddenSize, outputSize
   self.layers = {}

   if self.reverseInput == false and self.reverseOutput == false then
      for l=1,L do
         local layerInputSize = l == 1 and D or R
         table.insert(
            self.layers, 
            ntg.network.LSTM(layerInputSize, H, false, false))
      end
   elseif self.reverseInput == false and self.reverseOutput == true then
      for l=1,L do
         local layerInputSize = l == 1 and D or R
         local revOut = false
         if l == L then revOut = true end
         table.insert(
            self.layers, 
            ntg.network.LSTM(layerInputSize, H, false, revOut))
      end
   elseif self.reverseInput == true and self.reverseOutput == false then
      for l=1,L do
         local layerInputSize = l == 1 and D or R
         local revIn = false
         if l == 1 then
            revIn = true
         end
         table.insert(
            self.layers, 
            ntg.network.LSTM(layerInputSize, H, revIn, false))
      end
   elseif self.reverseInput == true and self.reverseOutput == true then
      for l=1,L do
         local layerInputSize = l == 1 and D or R
         local revIn = false
         if l == 1 then
            revIn = true
         end
         local revOut = false
         if l == L then
            revOut = true
         end

         table.insert(
            self.layers, 
            ntg.network.LSTM(layerInputSize, H, revIn, revOut))
      end

   end

end

function StackedLSTM:reset()
   for layer=1,#self.layers do
      self.layers[layer]:reset()
   end
end

function StackedLSTM:setParametersCopy(otherParams)
   local myParams, myGradParams = self:parameters()
   assert(#myParams == #otherParams, "Incompatible params")
   for i=1,#myParams do
      myParams[i]:copy(otherParams[i])
   end
   return self
end

function StackedLSTM:zeroGradParameters()
   for i=1,#self.layers do
      self.layers[i]:zeroGradParameters()
   end
   return self
end

function StackedLSTM:parameters()
   local params = {}
   local gradParams = {}

   for layer=1,#self.layers do
      local lParams, lGradParams = self.layers[layer]:parameters()
      ntg.util.Table.append(params, lParams)
      ntg.util.Table.append(gradParams, lGradParams)
   end
   return params, gradParams
end

function StackedLSTM:getOutputState(index)
   local state = {}
   if index and self.reverseOutput then
      self._index = self._index or torch.Tensor()
      self._index = self._index:typeAs(self.output)
         :resize(self.output:size(2)):fill(self.output:size(1))
         :csub(index):add(1)
   end

   for layer=1,self.layerSize do
      if index and self.reverseOutput and layer < self.layerSize then
         table.insert(state, self.layers[layer]:getOutputState(self._index))
      else
         table.insert(state, self.layers[layer]:getOutputState(index))
     end
   end
   return state
end


function StackedLSTM:getCellState(index)
   local state = {}

   if index and self.reverseOutput then
      self._index = self._index or torch.Tensor()
      self._index = self._index:typeAs(self.output)
         :resize(self.output:size(2)):fill(self.output:size(1))
         :csub(index):add(1)
      
   end

   for layer=1,self.layerSize do
      if index and self.reverseOutput and layer < self.layerSize then
         table.insert(state, self.layers[layer]:getCellState(self._index))
      else
         table.insert(state, self.layers[layer]:getCellState(index))
     end
   end
   return state
end

function StackedLSTM:setOutputGradState(gradOutputState, index)
   for l=1,self.layerSize do
      if index and self.reverseOutput and l < self.layerSize then
         self.layers[l]:setOutputGradState(gradOutputState[l], self._index)
      else
         self.layers[l]:setOutputGradState(gradOutputState[l], index)
     end
   end
end

function StackedLSTM:setCellGradState(gradCellState, index)
   for l=1,self.layerSize do
      if index and self.reverseOutput and l < self.layerSize then
         self.layers[l]:setCellGradState(gradCellState[l], self._index)
      else
         self.layers[l]:setCellGradState(gradCellState[l], index)
     end
   end
end




function StackedLSTM:forward(input, mask)

   local revMask
   if mask and self.reverseInput then
      self._revMaskNet = self._revMaskNet or nn.SeqReverseSequence(1)
      self._revMaskNet = self._revMaskNet:type(mask:type())
      revMask = self._revMaskNet:forward(mask)
   end

   self.inputs = {}
   local nextInput = input
   for layer=1,#self.layers do
      table.insert(self.inputs, nextInput)
      if revMask and layer > 1 then
         nextInput = self.layers[layer]:forward(nextInput, revMask)
      else
         nextInput = self.layers[layer]:forward(nextInput, mask)
      end
   end
   self.output = nextInput
   return self.output
end

function StackedLSTM:backward(input, gradOutput, mask)
   local revMask
   if mask and self.reverseInput then
      revMask = self._revMaskNet.output
   end
   local nextGradOutput = gradOutput
   for layer=#self.layers, 1, -1 do
      local layerInput = layer == 1 and input or self.layers[layer - 1].output 
      if revMask and layer > 1 then
         nextGradOutput = self.layers[layer]:backward(
            layerInput, nextGradOutput, revMask)
      else
         nextGradOutput = self.layers[layer]:backward(
            layerInput, nextGradOutput, mask)
      end
   end
   self.gradInput = nextGradOutput
   return self.gradInput
end

function StackedLSTM:description(offset)
   offset = offset or ''
   local buffer = offset .. 'Network: StackedLSTM  -------------------+\n' 
      .. offset .. '|          layers: ' .. #self.layers .. '\n' 
      .. offset .. '|      input size: ' .. self.inputSize .. '\n' 
      .. offset .. '|       cell size: ' .. self.hiddenSize .. '\n'
      .. offset .. '|     output size: ' .. self.outputSize .. '\n'
      .. offset .. '|    total params: ' .. 
         (self.layers[1].weight:nElement() + self.layers[1].bias:nElement()) * #self.layers .. '\n'
      .. offset .. '+-------------------------------------+' 
   return buffer
end

function StackedLSTM:setInitOutputState(state)
   assert(#state == self.layerSize)
   for i=1,#state do
      self.layers[i]:setInitOutputState(state[i])
   end
end

function StackedLSTM:setInitCellState(state)
   assert(#state == self.layerSize)
   for i=1,#state do
      self.layers[i]:setInitCellState(state[i])
   end
end

function StackedLSTM:getGradInitOutputState()
   self.grad_init_output_state = self.grad_init_output_state or {}
   for i=1,#self.layers do
      self.grad_init_output_state[i] = self.layers[i]:getGradInitOutputState()
   end
   return self.grad_init_output_state
end

function StackedLSTM:getGradInitCellState()
   self.grad_init_cell_state = self.grad_init_cell_state or {}
   for i=1,#self.layers do
      self.grad_init_cell_state[i] = self.layers[i]:getGradInitCellState()
   end
   return self.grad_init_cell_state
end
