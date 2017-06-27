local LSTM, parent = torch.class('ntg.network.LSTM', 'ntg.network.Network')

function LSTM:__init(inputSize, hiddenSize, reverseInput, reverseOutput)

   self.reverseInput = reverseInput
   if self.reverseInput == nil then self.reverseInput = false end
   self.reverseOutput = reverseOutput
   if self.reverseOutput == nil then self.reverseOutput = false end
   
   local outputSize = hiddenSize
   local D, H, R = inputSize, hiddenSize, outputSize
   self.inputSize, self.hiddenSize, self.outputSize = D, H, R
   
   self.weight = torch.Tensor(D+R, 4 * H)
   self.gradWeight = torch.Tensor(D+R, 4 * H)
   
   self.bias = torch.Tensor(4 * H)
   self.gradBias = torch.Tensor(4 * H):zero()

   self:reset()

   self.cell = torch.Tensor()    -- This will be  (T, N, H)
   self.gates = torch.Tensor()   -- This will be (T, N, 4H)
   self.buffer1 = torch.Tensor() -- This will be (N, H)
   self.buffer2 = torch.Tensor() -- This will be (N, H)
   self.buffer3 = torch.Tensor() -- This will be (1, 4H)
   self.grad_a_buffer = torch.Tensor() -- This will be (N, 4H)

   self.h0 = torch.Tensor()
   self.c0 = torch.Tensor()

   self._remember = 'neither'

   self.grad_c0 = torch.Tensor()
   self.grad_h0 = torch.Tensor()
   self.grad_x = torch.Tensor()
   self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}

   
end

function LSTM:description(offset)
   offset = offset or ''
   local buffer = offset .. 'Network: LSTM  -----------------------+\n' 
      .. offset .. '|      input size: ' .. self.inputSize .. '\n' 
      .. offset .. '|       cell size: ' .. self.hiddenSize .. '\n'
      .. offset .. '|     output size: ' .. self.outputSize .. '\n'
      .. offset .. '|    total params: ' .. 
         self.weight:nElement() + self.bias:nElement() .. '\n'
      .. offset .. '+-------------------------------------+' 
   return buffer
end

function LSTM:reset(std)
   if not std then
      std = 1.0 / math.sqrt(self.outputSize + self.inputSize)
   end
   self.bias:zero()
   self.bias[{{self.outputSize + 1, 2 * self.outputSize}}]:fill(1)
   self.weight:normal(0, std)
   return self
end

function LSTM:type(typeString)
   
   self.weight = self.weight:type(typeString)
   self.gradWeight = self.gradWeight:type(typeString)
   
   self.bias = self.bias:type(typeString)
   self.gradBias = self.gradBias:type(typeString)

   self.cell = self.cell:type(typeString)
   self.gates = self.gates:type(typeString)
   self.buffer1 = self.buffer1:type(typeString) 
   self.buffer2 = self.buffer2:type(typeString) 
   self.buffer3 = self.buffer3:type(typeString)
   self.grad_a_buffer = self.grad_a_buffer:type(typeString) 

   self.h0 = self.h0:type(typeString)
   self.c0 = self.c0:type(typeString)

   self.grad_c0 = self.grad_c0:type(typeString)
   self.grad_h0 = self.grad_h0:type(typeString)
   self.grad_x = self.grad_x:type(typeString)
   self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
 
end

function LSTM:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
   return self
end

function LSTM:parameters()
   local params = {self.weight, self.bias}
   local gradParams = {self.gradWeight, self.gradBias}
   return params, gradParams
end

function LSTM:getCellState(step)
   if step == nil then
      return self.cell[-1]
   elseif torch.type(step) == 'number' then
      return self.cell[step]
   else
      assert(step:dim() == 1)
      assert(step:size(1) == self.cell:size(2))
      local N = step:size(1)
      local H = self.hiddenSize
      self._cellStateBuffer = self._cellStateBuffer or torch.Tensor()
      self._cellStateBuffer = self._cellStateBuffer:typeAs(self.cell)
      local cell = self._cellStateBuffer:resize(N, H)
      for i=1,N do
         cell[i]:copy(self.cell[step[i]][i])
      end
      return cell
   end
end

function LSTM:getOutputState(step)
   if step == nil then
      return self.output[-1]
   elseif torch.type(step) == 'number' then
      return self.output[step]
   else
      assert(step:dim() == 1)
      assert(step:size(1) == self.output:size(2))
      local N = step:size(1)
      local R = self.outputSize
      self._outputStateBuffer = self._outputStateBuffer or torch.Tensor()
      self._outputStateBuffer = self._outputStateBuffer:typeAs(self.cell)
      local output = self._outputStateBuffer:resize(N, R)
      for i=1,N do
         output[i]:copy(self.output[step[i]][i])
      end
      return output
   end
end




function LSTM:setInitCellState(cellState)
   assert(cellState:dim() == 2, 'LSTM cell state must be a 2d tensor.')
   assert(cellState:size(2) == self.hiddenSize, 
      'Bad LSTM cell state dimension.')
   self._initCellState = cellState
end  

function LSTM:getGradInitCellState()
   return self.gradInitCellState
end

function LSTM:getGradInitOutputState()
   return self.gradInitOutputState
end

function LSTM:setInitOutputState(outputState)
   assert(outputState:dim() == 2, 'LSTM output state must be a 2d tensor.')
   assert(outputState:size(2) == self.outputSize, 
      'Bad LSTM output state dimension.')
   self._initOutputState = outputState
end  



local function check_dims(x, dims)
   assert(x:dim() == #dims)
   for i, d in ipairs(dims) do
      assert(x:size(i) == d)
   end
end

-- makes sure x, h0, c0 and gradOutput have correct sizes.
-- batchfirst = true will transpose the N x T to conform to T x N
function LSTM:_prepare_size(input, gradOutput)
   local c0, h0, x
   if torch.type(input) == 'table' and #input == 3 then
      c0, h0, x = unpack(input)
   elseif torch.type(input) == 'table' and #input == 2 then
      h0, x = unpack(input)
   elseif torch.isTensor(input) then
      x = input
   else
      assert(false, 'invalid input')
   end
   assert(x:dim() == 3, "Only supports batch mode")
   
   local T, N = x:size(1), x:size(2)
   local H, D = self.outputSize, self.inputSize
   
   check_dims(x, {T, N, D})
   if h0 then
      check_dims(h0, {N, H})
   end
   if c0 then
      check_dims(c0, {N, H})
   end
   if gradOutput then
      check_dims(gradOutput, {T, N, H})
   end
   return c0, h0, x, gradOutput
end
 
--[[
Input:
- c0: Initial cell state, (N, H)
- h0: Initial hidden state, (N, H)
- x: Input sequence, (T, N, D)  
Output:
- h: Sequence of hidden states, (T, N, H)
--]]

function LSTM:forward(input, mask)
   self.recompute_backward = true
   local c0, h0, x = self:_prepare_size(input)
   local N, T = x:size(2), x:size(1)
   local H, R, D = self.hiddenSize, self.outputSize, self.inputSize
   
   if mask then 
      assert(mask:dim() == 2)
      assert(mask:size(1) == T)
      assert(mask:size(2) == N)
   end
   self._output = self._output or self.weight.new()
   
   --self._return_grad_c0 = (c0 ~= nil)
   --self._return_grad_h0 = (h0 ~= nil)
   if not c0 then
      c0 = self.c0
      if self._initCellState then
         local prev_N = self._initCellState:size(1)
         assert(prev_N == N, 
            'Batch sizes must be consistent with _initCellState')
         c0:resizeAs(self._initCellState):copy(self._initCellState)
      else --c0:nElement() == 0 or not remember then
         c0:resize(N, H):zero()
      --elseif remember then
      --   local prev_T, prev_N = self.cell:size(1), self.cell:size(2)
      --   assert(prev_N == N, 'batch sizes must be constant to remember states')
      --   c0:copy(self.cell[prev_T])
      end
   end
   if not h0 then
      h0 = self.h0
      if self._initOutputState then
         local prev_N = self._initOutputState:size(1)
         assert(prev_N == N, 
            'Batch sizes must be consistent with _initOutputState')
         h0:resizeAs(self._initOutputState):copy(self._initOutputState)
      else --if h0:nElement() == 0 or not remember then
         h0:resize(N, R):zero()
      --elseif remember then
      --   local prev_T, prev_N = self._output:size(1), self._output:size(2)
      --   assert(prev_N == N, 'batch sizes must be the same to remember states')
      --   h0:copy(self._output[prev_T])
      end
   end

   local bias_expand = self.bias:view(1, 4 * H):expand(N, 4 * H)

   local Wx = self.weight:narrow(1,1,D)
   local Wh = self.weight:narrow(1,D+1,R)

   local h, c = self._output, self.cell
   h:resize(T, N, R):zero()
   c:resize(T, N, H):zero()
   local prev_h, prev_c = h0, c0
   self.gates:resize(T, N, 4 * H):zero()

   local start, stop, step = 1,T,1
   if self.reverseInput then
      start, stop, step = T,1,-1
   end

   self.gradOutputBuffers = {}
   self.gradCellBuffers = {}
   for t = start,stop,step do
      local cur_x = x[t]

      local t_h
      if self.reverseInput then
         if self.reverseOutput then
            t_h = t
         else
            t_h = T - t + 1
         end  
      else
         if self.reverseOutput then
            t_h = T - t + 1
         else
            t_h = t
         end
      end

      self.next_h = h[t_h]
      local next_c = c[t_h]
      local cur_gates = self.gates[t_h]
      cur_gates:addmm(bias_expand, cur_x, Wx)
      cur_gates:addmm(prev_h, Wh)
      cur_gates[{{}, {1, 3 * H}}]:sigmoid()
      cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
      local i = cur_gates[{{}, {1, H}}] -- input gate
      local f = cur_gates[{{}, {H + 1, 2 * H}}] -- forget gate
      local o = cur_gates[{{}, {2 * H + 1, 3 * H}}] -- output gate
      local g = cur_gates[{{}, {3 * H + 1, 4 * H}}] -- input transform
      self.next_h:cmul(i, g)
      next_c:cmul(f, prev_c):add(self.next_h)
      self.next_h:tanh(next_c):cmul(o)

      if mask then
         local cur_mask = mask[t]
         local cellMask = cur_mask:view(N, 1):expand(N, H)
         next_c:maskedFill(cellMask, 0)
         local outputMask = cur_mask:view(N, 1):expand(N, R)
         self.next_h:maskedFill(outputMask, 0)
         local gatesMask = cur_mask:view(N, 1):expand(N, 4 * H)
         cur_gates:maskedFill(gatesMask, 0)
      end      
      
      prev_h, prev_c = self.next_h, next_c
   end
   self._initOutputState = nil
   self._initCellState = nil
  
   self.output = self._output
 
   return self.output
end

function LSTM:setOutputGradState(gradOutputState, positions)

   if positions == nil then
      --if self.reverseInput then
       --  self.gradOutputBuffers[1] = gradOutputState
      --else
      self.gradOutputBuffers[self.output:size(1)] = gradOutputState
      --end
      return self
   end

   local batchSize = gradOutputState:size(1)
   local dimSize = gradOutputState:size(2)

   self.gradBuffers1 = self.gradBuffers1 or {}
   local bp = 1

   self.gradOutputBuffers = {}
   for i=1,positions:size(1) do
      local p = positions[i]
      local buffer = self.gradOutputBuffers[p]
      if buffer then
         buffer[i]:copy(gradOutputState[i])
      else
         buffer = self.gradBuffers1[bp] or torch.Tensor()
         buffer = buffer:typeAs(gradOutputState)
            :resize(batchSize, dimSize):zero()
         buffer[i]:copy(gradOutputState[i])
         self.gradOutputBuffers[p] = buffer
         self.gradBuffers1[bp] = buffer
         bp = bp + 1
      end
   end
   return self
end

function LSTM:setCellGradState(gradCellState, positions)

   if positions == nil then
      if self.reverseInput then
         self.gradCellBuffers[1] = gradCellState
      else
         self.gradCellBuffers[self.output:size(1)] = gradCellState
      end
      return self
   end


   local batchSize = gradCellState:size(1)
   local dimSize = gradCellState:size(2)

   self.gradBuffers2 = self.gradBuffers2 or {}
   local bp = 1

   self.gradCellBuffers = {}
   for i=1,positions:size(1) do
      local p = positions[i]
      local buffer = self.gradCellBuffers[p]
      if buffer then
         buffer[i]:copy(gradCellState[i])
      else
         buffer = self.gradBuffers2[bp] or torch.Tensor()
         buffer = buffer:typeAs(gradCellState)
            :resize(batchSize, dimSize):zero()
         buffer[i]:copy(gradCellState[i])
         self.gradCellBuffers[p] = buffer
         self.gradBuffers2[bp] = buffer
         bp = bp + 1
      end
   end
   return self
end

function LSTM:backward(input, gradOutput, mask)
   
   local c0, h0, x, grad_h = self:_prepare_size(input, gradOutput)
   assert(grad_h, "Expecting gradOutput")
   local N, T = x:size(2), x:size(1)
   local H, R, D = self.hiddenSize, self.outputSize, self.inputSize
   
   self._grad_x = self._grad_x or self.weight:narrow(1,1,D).new()
   
   if not c0 then c0 = self.c0 end
   if not h0 then h0 = self.h0 end

   local grad_c0, grad_h0, grad_x = self.grad_c0, self.grad_h0, self._grad_x
   local h, c = self._output, self.cell
   
   local Wx = self.weight:narrow(1,1,D)
   local Wh = self.weight:narrow(1,D+1,R)
   local grad_Wx = self.gradWeight:narrow(1,1,D)
   local grad_Wh = self.gradWeight:narrow(1,D+1,R)
   local grad_b = self.gradBias

   grad_h0:resizeAs(h0):zero()
   grad_c0:resizeAs(c0):zero()
   grad_x:resizeAs(x):zero()
   self.buffer1:resizeAs(h0)
   self.buffer2:resizeAs(c0)
   self.grad_next_h = self.buffer1:zero()
   local grad_next_c = self.buffer2:zero()
 
   local start, stop, step = T,1,-1
   if self.reverseInput then
      start, stop, step = 1,T,1
   end

  
   for t = start,stop,step do
      local t_h, tm1_h
      if self.reverseInput then
         if self.reverseOutput then
            t_h = t
            tm1_h = t+1
         else
            t_h = T - t + 1
            tm1_h = T - t 
         end  
      else
         if self.reverseOutput then
            t_h = T - t + 1
            tm1_h = T-t+2
         else
            t_h = t
            tm1_h = t - 1
         end
      end

      local next_h, next_c = h[t_h], c[t_h]
      local prev_h, prev_c = nil, nil
      if t == stop then
         prev_h, prev_c = h0, c0
      else
         prev_h, prev_c = h[tm1_h], c[tm1_h]
      end
      self.grad_next_h:add(grad_h[t_h])
      if self.gradOutputBuffers[t_h] then
         self.grad_next_h:add(self.gradOutputBuffers[t_h])
      end
      if self.gradCellBuffers[t_h] then
         grad_next_c:add(self.gradCellBuffers[t_h])
      end


      if mask then
         local cur_mask = mask[t]
         local outputMask = cur_mask:view(N, 1):expand(N, R)
         self.grad_next_h:maskedFill(outputMask, 0)
      end      
 
--      if self.maskzero and torch.type(self) ~= 'nn.SeqLSTM' then 
--         -- we only do this for sub-classes (LSTM doesn't need it)   
--         -- build mask from input
--         local cur_x = x[t]
--         local vectorDim = cur_x:dim()
--         self._zeroMask = self._zeroMask or cur_x.new()
--         self._zeroMask:norm(cur_x, 2, vectorDim)
--         self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaByteTensor() or torch.ByteTensor())
--         self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)
--         -- zero masked gradOutput
--         self:recursiveMask(self.grad_next_h, self.zeroMask)
--      end
      
      -- for LSTMP
--      self:gradAdapter(scale, t)

      local i = self.gates[{t_h, {}, {1, H}}]
      local f = self.gates[{t_h, {}, {H + 1, 2 * H}}]
      local o = self.gates[{t_h, {}, {2 * H + 1, 3 * H}}]
      local g = self.gates[{t_h, {}, {3 * H + 1, 4 * H}}]
      
      local grad_a = self.grad_a_buffer:resize(N, 4 * H):zero()
      local grad_ai = grad_a[{{}, {1, H}}]
      local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
      local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
      local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
      
      -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
      -- to to compute grad_next_c. We will need tanh_next_c (stored in grad_ai)
      -- to compute grad_ao; the other values can be overwritten after we compute
      -- grad_next_c
      local tanh_next_c = grad_ai:tanh(next_c)
      local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c)
      local my_grad_next_c = grad_ao
      my_grad_next_c:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(self.grad_next_h)
      grad_next_c:add(my_grad_next_c)

      
      -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after
      -- that we can overwrite it.
      grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(self.grad_next_h)

      -- Use grad_ai as a temporary buffer for computing grad_ag
      local g2 = grad_ai:cmul(g, g)
      grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_next_c)

      -- We don't need any temporary storage for these so do them last
      grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_next_c)
      grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_next_c)
      
      grad_x[t]:mm(grad_a, Wx:t())
      grad_Wx:addmm(1, x[t]:t(), grad_a)
      grad_Wh:addmm(1, prev_h:t(), grad_a)
      local grad_a_sum = self.buffer3:resize(1, 4 * H):sum(grad_a, 1)
      grad_b:add(1, grad_a_sum)
      
      self.grad_next_h = torch.mm(grad_a, Wh:t())
      grad_next_c:cmul(f)
      
   end
   grad_h0:copy(self.grad_next_h)
   grad_c0:copy(grad_next_c)
   
   self.grad_x = grad_x
   
--   self.gradPrevOutput = nil
--   self.userNextGradCell = nil
   self.gradInitCellState = self.grad_c0
   self.gradInitOutputState = self.grad_h0
   
--   if self._return_grad_c0 and self._return_grad_h0 then
--      self.gradInput = {self.grad_c0, self.grad_h0, self.grad_x}
--   elseif self._return_grad_h0 then
--      self.gradInput = {self.grad_h0, self.grad_x}
--   else
      self.gradInput = self.grad_x
--   end

   return self.gradInput
end
