local MLPPredictor = torch.class('ntg.network.MLPPredictor')

function MLPPredictor:__init(input_size, vocab_size, layer_size, activation)

   self.input_size = input_size
   self.vocab_size = vocab_size
   self.layer_size = layer_size or 0
   self.activation = activation or 'Tanh'
   assert(self.input_size > 0)
   assert(self.vocab_size > 0)
   assert(self.layer_size >= 0)

   self.mlp_weight = torch.Tensor(self.layer_size, input_size, input_size)
   self.mlp_bias = torch.Tensor(self.layer_size, input_size)
   self.mlp_grad_weight = torch.Tensor(self.layer_size, input_size, input_size)
   self.mlp_grad_bias = torch.Tensor(self.layer_size, input_size)
   self.mlp_activations = torch.Tensor()
   self.mlp_transfer_functions = {}
   for i=1,self.layer_size do
      table.insert(self.mlp_transfer_functions, nn[self.activation]())
   end
   
   self.logits_weight = torch.Tensor(vocab_size, input_size)
   self.logits_bias = torch.Tensor(vocab_size)
   self.logits_grad_weight = torch.Tensor(vocab_size, input_size)
   self.logits_grad_bias = torch.Tensor(vocab_size)
   self.logits = torch.Tensor()
   self.log_soft_max = nn.LogSoftMax()
    
   self.grad_input = torch.Tensor()

   self:reset()
end

function MLPPredictor:reset()
   self.mlp_weight:normal()
   self.mlp_bias:normal()
   self.logits_weight:normal()
   self.logits_bias:normal()
end

function MLPPredictor:zeroGradParameters()
   self.mlp_grad_weight:zero()
   self.mlp_grad_bias:zero()
   self.logits_grad_weight:zero()
   self.logits_grad_bias:zero()
end

function MLPPredictor:parameters()
   local params = {}
   local grad_params = {}
   table.insert(params, self.mlp_weight)
   table.insert(params, self.mlp_bias)
   table.insert(params, self.logits_weight)
   table.insert(params, self.logits_bias)
   table.insert(grad_params, self.mlp_grad_weight)
   table.insert(grad_params, self.mlp_grad_bias)
   table.insert(grad_params, self.logits_grad_weight)
   table.insert(grad_params, self.logits_grad_bias)
   return params, grad_params
end

function MLPPredictor:forward(input, mask)
   local T = input:size(1)
   local N = input:size(2)
   local TN = T * N
   local L = self.layer_size
   local H = self.input_size
   local V = self.vocab_size

   self.mlp_activations:resize(L, TN, H)

   local next_input = input:view(TN, H)

   for i=1,self.layer_size do
      local W = self.mlp_weight[i]
      local b = self.mlp_bias[i]:view(1, H):expand(TN, H)
      local act = self.mlp_activations[i]:addmm(b, next_input, W:t())
      next_input = self.mlp_transfer_functions[i]:forward(act)
   end

   local logits = self.logits:resize(TN, V)
   local W = self.logits_weight
   local b = self.logits_bias:view(1, V):expand(TN, V)
   logits:addmm(b, next_input, W:t())
   self.output = self.log_soft_max:forward(logits)

   self.output = self.output:view(T, N, V)
   if mask then
      local mask3d = mask:view(T, N, 1):expand(T, N, V)
      self.output:maskedFill(mask3d, 0)
   end

   return self.output
end


function MLPPredictor:backward(input, grad_output, mask)

   local T = input:size(1)
   local N = input:size(2)
   local H = self.input_size
   local V = self.vocab_size
   local TN = T * N

   if mask then    
      local mask3d = mask:view(T, N, 1):expand(T, N, V)
      grad_output:maskedFill(mask3d, 0)
   end

   local grad_output_flat = grad_output:view(TN, V)
   local grad_lsm = self.log_soft_max:backward(self.logits, grad_output_flat)

   local logit_in = self.mlp_transfer_functions[self.layer_size].output
   self.logits_grad_weight:addmm(grad_lsm:t(), logit_in)
   self.logits_grad_bias = self.logits_grad_bias:sum(grad_lsm, 1):squeeze()
   self.grad_input:resize(TN, H):mm(grad_lsm, self.logits_weight)

   for i=self.layer_size,1,-1 do
      local tf = self.mlp_transfer_functions[i]
      local act = self.mlp_activations[i]
      local grad_tf = tf:backward(act, self.grad_input)
      local W = self.mlp_weight[i]
      local dW = self.mlp_grad_weight[i]
      local db = self.mlp_grad_bias[i]
      local layer_input = i == 1 and input:view(TN, H) 
            or self.mlp_transfer_functions[i-1].output
      dW:addmm(grad_tf:t(), layer_input)
      db:sum(grad_tf, 1):squeeze() 
      self.grad_input:resize(TN, H):mm(grad_tf, W)
   end

   self.grad_input = self.grad_input:view(T, N, H)

   return self.grad_input
--   os.exit()
--
--   local input_flat3d = input_flat:view(flat_size, 1, self.input_size)
--
--   local grad_layer = self.output_transform:backward(
--      self.activation_layers[#self.activation_layers].output, gradOutput)
--
--   for i=#self.activation_layers,1,-1 do
--      local a = self.hidden_layers[i]
--      local grad_act = self.activation_layers[i]:backward(a, grad_layer)
--
--
--      local W = self.weight[i]
--      local W3d = W:expand(flat_size, W:size(2), W:size(3))
--
--      self.grad_bias[i]:add(grad_act:sum(1))
--
--      local input_layer
--      if i == 1 then 
--         input_layer = input_flat3d 
--      else
--         input_layer = self.activation_layers[i-1].output
--      end
--
--      grad_layer = torch.bmm(grad_act, W3d:transpose(3,2))
--
--      local batch_grad_weight = torch.bmm(
--         input_layer:transpose(3,2), grad_act)
--      self.grad_weight[i]:add(batch_grad_weight:sum(1))
--   end
--
--   self.grad_input = self.input_transform:backward(input, grad_layer)
--   return self.grad_input

end

function MLPPredictor:float()

   --self.linear_outputs = self.linear_outputs:float()
   self.logits = self.logits:float()
   self.input_transform = self.input_transform:float()
   self.output_transform = self.output_transform:float()
   for i=1,#self.activation_layers do
      self.weight[i] = self.weight[i]:float()
      self.bias[i] = self.bias[i]:float()
      self.activation_layers[i] = self.activation_layers[i]:float()

   end
end
