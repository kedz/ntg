
local ConcatAttention = torch.class('ntg.network.ConcatAttention')

function ConcatAttention:__init(context_size, target_size, hidden_size)

    self.hidden_size = hidden_size
    self.context_weight = torch.Tensor(hidden_size, context_size)
    self.grad_context_weight = torch.Tensor(hidden_size, hidden_size)
    self.target_weight = torch.Tensor(hidden_size, target_size)
    self.grad_target_weight = torch.Tensor(hidden_size, hidden_size)
    self.attention_weight = torch.Tensor(1, hidden_size)
    self.grad_attention_weight = torch.Tensor(1, hidden_size)

    self.tanh = nn.Tanh()

    self.softmax = ntg.network.SoftMax()

   self._mask = torch.ByteTensor()

    self.concat_proj = torch.Tensor()
    self.context_proj = torch.Tensor()
    self.target_proj = torch.Tensor()
    self.logits = torch.Tensor()

    self.buffer1 = torch.Tensor()
    self.buffer2 = torch.Tensor()

   self:reset()
end

function ConcatAttention:zeroGradParameters()
   self.grad_attention_weight:zero()
   self.grad_context_weight:zero()
   self.grad_target_weight:zero()

end

function ConcatAttention:reset()
   self.context_weight:normal()
   self.target_weight:normal()
   self.attention_weight:normal()
end

function ConcatAttention:forward(context, target, context_mask, target_mask)

   local Tc = context:size(1)
   local N = context:size(2)
   local Hc = context:size(3)
   local TcN = Tc * N

   local Tt = target:size(1)
   local Ht = target:size(3)
   local H = self.hidden_size
   local TtN = Tt * N

   local concat_proj = self.concat_proj:resize(N, Tt, Tc, H) 
   local context_proj = self.context_proj:resize(TcN, H) 
   local target_proj = self.target_proj:resize(TtN, H) 

   context_proj:mm(context:view(TcN, H), self.context_weight:t())
   target_proj:mm(target:view(TtN, H), self.target_weight:t())
   concat_proj:add(
      context_proj:view(
         Tc, N, H):view(Tc, 1, N, H):transpose(3,1):expand(N, Tt, Tc, H),
      target_proj:view(
         Tt, N, H):view(Tt, N, 1, H):transpose(2,1):expand(N, Tt, Tc, H))
   
   self.tanh:forward(concat_proj)

   local logits = self.logits:resize(N * Tt * Tc, 1)
   logits:mm(self.tanh.output:view(N * Tt *Tc, H), 
      self.attention_weight:t())

   local mask = nil
   if context_mask or target_mask then
      mask = self._mask:resize(N, Tt, Tc):zero()
   end
   if context_mask then
      mask:add(context_mask:view(Tc, 1, N):permute(3,2,1):expand(N, Tt, Tc))
   end
   if target_mask then
      mask:add(target_mask:view(Tt, 1, N):permute(3, 1, 2):expand(N, Tt, Tc))
   end
   if mask then mask:cmin(1) end
   self.output = self.softmax:forward(logits:view(N, Tt, Tc), mask)
  
   return self.output

end



function ConcatAttention:backward(context, target, grad_output, 
      mask_context, mask_target)

   local Tc = context:size(1)
   local N = context:size(2)
   local Hc = context:size(3)
   local TcN = Tc * N

   local Tt = target:size(1)
   local Ht = target:size(3)
   local H = self.hidden_size
   local TtN = Tt * N

   self.grad_logits = self.grad_logits or torch.Tensor()
   self.grad_logits = self.grad_logits:typeAs(self.logits):resize(N*Tc *Tt, H)
   self.grad_input1 = self.grad_input1 or torch.Tensor()
   self.grad_input1 = self.grad_input1:typeAs(context):resizeAs(context)
   self.grad_input2 = self.grad_input2 or torch.Tensor()
   self.grad_input2 = self.grad_input2:typeAs(context):resizeAs(context)


   local mask = nil
   if mask_context or mask_target then
      mask = self._mask
   end


   local grad_softmax = self.softmax:backward(
      self.logits:view(N, Tt, Tc), grad_output, mask)

   self.grad_attention_weight:addmm(
      grad_softmax:view(N * Tc * Tt, 1):t(),
      self.tanh.output:view(N * Tt * Tc, H))

   local grad_logits = self.grad_logits:mm(
      grad_softmax:view(N * Tc * Tt, 1),
      self.attention_weight)

   local grad_tanh = self.tanh:backward(self.concat_proj, 
      grad_logits:view(N, Tt, Tc, H))

   local grad_context_repl = torch.mm(
      self.buffer1:resize(N * Tt * Tc, H),
      grad_tanh:view(N * Tt * Tc,  H), 
      self.context_weight):view(N, Tt, Tc, H)

   self.grad_input1 = self.grad_input1:sum( 
      grad_context_repl, 2):squeeze():permute(2, 1, 3)

   local grad_target_repl = torch.mm(
      self.buffer1:resize(N * Tt * Tc, H),
      grad_tanh:view(N * Tt * Tc,  H), 
      self.target_weight):view(N, Tt, Tc, H)

   self.grad_input2 = self.grad_input2:sum( 
      grad_target_repl, 3):squeeze():permute(2, 1, 3)

   local context_repl = self.buffer2:resize(H, N, Tt, Tc):copy(
      context:view(Tc, 1, N, H):permute(4, 3, 2, 1):expand(H, N, Tt, Tc))
   self.grad_context_weight:addmm(
      grad_tanh:view(N * Tt * Tc, H):t(),
      context_repl:view(H, N * Tt * Tc):t())

   local target_repl = self.buffer2:resize(H, N, Tt, Tc):copy(
      target:view(Tt, 1, N, H):permute(4, 3, 1, 2):expand(H, N, Tt, Tc))

   self.grad_target_weight:addmm(
      grad_tanh:view(N * Tt * Tc, H):t(),
      target_repl:view(H, N * Tt * Tc):t())

   return self.grad_input1, self.grad_input2
end
