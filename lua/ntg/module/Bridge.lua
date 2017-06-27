
local Bridge = torch.class('ntg.module.Bridge')

function Bridge:__init(stateOp, projOp, inputDim, outputDim, layers, hasCell)  

   local network = layers > 1 and nn.ParallelTable() or nn.Sequential()
   --local network = nn.ParallelTable()
   for layer=1,layers do
      local net = nn.Sequential()
      if stateOp == 'identity' then
         net:add(nn.Identity())
      elseif stateOp == 'concat' then
         net:add(nn.JoinTable(2))
      elseif stateOp == 'sum' then
         net:add(nn.CAddTable())
      elseif stateOp == 'mean' then
         net:add(nn.CAddTable()):add(nn.MulConstant(.5, true))
      elseif stateOp == 'forward' then
         net:add(nn.SelectTable(1))
      else
         assert(stateOp == 'backward')
         net:add(nn.SelectTable(2))
      end
      if projOp == 'linear' then
         if stateOp == 'concat' then
            net:add(nn.Linear(inputDim * 2, outputDim))
         else
            net:add(nn.Linear(inputDim, outputDim))
         end
      end
      network:add(net)
   end

   if hasCell then
      self.network = nn.ParallelTable():add(network):add(network:clone())

    --  if layers == 1 and stateOp ~= 'identity' then
    --     self.network = nn.Sequential():add(self.network)
    --        :add(nn.FlattenTable())
    --  end
   else
      self.network = network
   end

   self.network:reset()

end

function Bridge:forward(encoderState)
   self.decoderState = self.network:forward(encoderState)
   return self.decoderState
end

function Bridge:backward(encoderState, gradOutput)
   self.gradInput = self.network:backward(encoderState, gradOutput)
   return self.gradInput
end

function Bridge:parameters()
   local params, gradParams = self.network:parameters()
   if params == nil then
      return {}, {}
   else
      return params, gradParams
   end  
end

function Bridge:zeroGradParameters()
   self.network:zeroGradParameters()
end

function Bridge:reset()
   self.network:reset()
end
