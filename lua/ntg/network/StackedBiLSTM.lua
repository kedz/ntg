local StackedBiLSTM, parent = torch.class('ntg.network.StackedBiLSTM', 
   'ntg.network.Network')

local IDENTITY_OP = 1
local CONCAT_OP = 2
local FORWARD_OP = 3
local BACKWARD_OP = 4
local MEAN_OP = 5
local SUM_OP = 6
local mergeOpIds = {identity=1, concat=2, forward=3, backward=4, mean=5, sum=6}

function StackedBiLSTM:__init(layerSize, inputSize, hiddenSize, mergeOp)

   mergeOp = mergeOp or 'concat'
   assert(layerSize > 0)
   assert(inputSize > 0)
   assert(hiddenSize > 0)
   assert(mergeOpIds[mergeOp])
   self.layerSize = layerSize
   self.inputSize = inputSize
   self.hiddenSize = hiddenSize
   self.mergeOp = mergeOp
   self._mergeOpId = mergeOpIds[mergeOp]

      self.forwardNetwork = ntg.network.StackedLSTM(
         layerSize, inputSize, hiddenSize, false, false)
      self.reverseNetwork = ntg.network.StackedLSTM(
         layerSize, inputSize, hiddenSize, true, true)

   if self._mergeOpId == IDENTITY_OP then
      self.mergeNetwork = nn.Identity()
   elseif self._mergeOpId == CONCAT_OP then
      self.mergeNetwork = nn.JoinTable(3)
   elseif self._mergeOpId == FORWARD_OP then
      self.mergeNetwork = nn.SelectTable(1)
   elseif self._mergeOpId == BACKWARD_OP then
      self.mergeNetwork = nn.SelectTable(2)
   elseif self._mergeOpId == MEAN_OP then
      self.mergeNetwork = nn.Sequential()
         :add(nn.CAddTable())
         :add(nn.MulConstant(.5, false))
   elseif self._mergeOpId == SUM_OP then
      self.mergeNetwork = nn.Sequential()
         :add(nn.CAddTable())
   end
   
end

function StackedBiLSTM:setParametersCopy(otherParams)
   local myParams, myGradParams = self:parameters()
   assert(#myParams == #otherParams, "Incompatible params")
   for i=1,#myParams do
      myParams[i]:copy(otherParams[i])
   end
   return self
end

function StackedBiLSTM:zeroGradParameters()
   self.forwardNetwork:zeroGradParameters()
   self.reverseNetwork:zeroGradParameters()
   return self
end

function StackedBiLSTM:parameters()
   local params = {}
   local gradParams = {}
   local fParams, fGradParams = self.forwardNetwork:parameters()
   local rParams, rGradParams = self.reverseNetwork:parameters()
   ntg.util.Table.append(params, fParams)
   ntg.util.Table.append(params, rParams)
   ntg.util.Table.append(gradParams, fGradParams)
   ntg.util.Table.append(gradParams, rGradParams)
   return params, gradParams
end

function StackedBiLSTM:forward(input, mask)

   self.fwdOutput = self.forwardNetwork:forward(input, mask)
   self.revOutput = self.reverseNetwork:forward(input, mask)
   self.output = self.mergeNetwork:forward({self.fwdOutput, self.revOutput})
   return self.output

end

function StackedBiLSTM:backward(input, gradOutput, mask)

   local gradMerge = self.mergeNetwork:backward(
      {self.fwdOutput, self.revOutput}, gradOutput)
   local gradForward = self.forwardNetwork:backward(input, gradMerge[1], mask)
   local gradReverse = self.reverseNetwork:backward(input, gradMerge[2], mask)
   -- no need to add again here. could set rnns to use the same gradInput 
   -- memory
   self.gradInput = gradForward:add(gradReverse)
   return self.gradInput

end

function StackedBiLSTM:getForwardOutputState(index)
   return self.forwardNetwork:getOutputState(index)
end

function StackedBiLSTM:getForwardCellState(index)
   return self.forwardNetwork:getCellState(index)
end

function StackedBiLSTM:getReverseOutputState(index)
   return self.reverseNetwork:getOutputState(index)
end

function StackedBiLSTM:getReverseCellState(index)
   return self.reverseNetwork:getCellState(index)
end

function StackedBiLSTM:getOutputState(fwdIndex, revIndex)
   self._index = self._index or torch.Tensor()
   self._index = self._index:typeAs(self.fwdOutput)
      :resize(self.fwdOutput:size(2)):fill(self.fwdOutput:size(1))
      :csub(revIndex):add(1)
 
   local state = {}
   for l=1,self.layerSize do
      local actualRevIndex = l == self.layerSize and revIndex or self._index
      table.insert(state, 
         {self.forwardNetwork.layers[l]:getOutputState(fwdIndex),
          self.reverseNetwork.layers[l]:getOutputState(actualRevIndex)})
   end
   if self.layerSize == 1 then
      return state[1]
   end
   return state
end

function StackedBiLSTM:setOutputGradState(gradOutputState, fwdIndex, revIndex)
   self._index = self._index or torch.Tensor()
   self._index = self._index:typeAs(self.fwdOutput)
      :resize(self.fwdOutput:size(2)):fill(self.fwdOutput:size(1))
      :csub(revIndex):add(1)
 
   if self.layerSize == 1 then
      local actualRevIndex = revIndex 
      self.forwardNetwork.layers[1]:setOutputGradState(
            gradOutputState[1], fwdIndex)
      self.reverseNetwork.layers[1]:setOutputGradState(
            gradOutputState[2], actualRevIndex)
   else
      for l=1,self.layerSize do
         local actualRevIndex = l == self.layerSize and revIndex or self._index
         self.forwardNetwork.layers[l]:setOutputGradState(
            gradOutputState[l][1], fwdIndex)
         self.reverseNetwork.layers[l]:setOutputGradState(
            gradOutputState[l][2], actualRevIndex)
      end
   end
end


function StackedBiLSTM:getCellState(fwdIndex, revIndex)
   self._index = self._index or torch.Tensor()
   self._index = self._index:typeAs(self.fwdOutput)
      :resize(self.fwdOutput:size(2)):fill(self.fwdOutput:size(1))
      :csub(revIndex):add(1)
 
   local state = {}
   for l=1,self.layerSize do
      local actualRevIndex = l == self.layerSize and revIndex or self._index
      table.insert(state, 
         {self.forwardNetwork.layers[l]:getCellState(fwdIndex),
          self.reverseNetwork.layers[l]:getCellState(actualRevIndex)})
   end
   if self.layerSize == 1 then
      return state[1]
   end
   return state
end

function StackedBiLSTM:setCellGradState(gradCellState, fwdIndex, revIndex)
   self._index = self._index or torch.Tensor()
   self._index = self._index:typeAs(self.fwdOutput)
      :resize(self.fwdOutput:size(2)):fill(self.fwdOutput:size(1))
      :csub(revIndex):add(1)

   if self.layerSize == 1 then
      local actualRevIndex = revIndex 
      self.forwardNetwork.layers[1]:setCellGradState(
         gradCellState[1], fwdIndex)
      self.reverseNetwork.layers[1]:setCellGradState(
         gradCellState[2], actualRevIndex)
   else
      for l=1,self.layerSize do
         local actualRevIndex = l == self.layerSize and revIndex or self._index
         self.forwardNetwork.layers[l]:setCellGradState(
            gradCellState[l][1], fwdIndex)
         self.reverseNetwork.layers[l]:setCellGradState(
            gradCellState[l][2], actualRevIndex)
      end
   end
end

function StackedBiLSTM:reset()
   self.forwardNetwork:reset()
   self.reverseNetwork:reset()
end
