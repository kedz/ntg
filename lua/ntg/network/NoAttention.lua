
local NoAttention = torch.class('ntg.network.NoAttention')

function NoAttention:forward(source_input, target_input, 
      source_mask, target_mask)
   self.output = target_input
   return self.output
end

function NoAttention:backward(source_input, target_input, grad_output, 
      source_mask, target_mask)
   self.grad_source = self.grad_source or torch.Tensor()
   self.grad_source = self.grad_source:typeAs(
      source_input):resizeAs(source_input):zero()
   self.grad_target = grad_output
   return self.grad_source, self.grad_target
end

function NoAttention:parameters()
   self.params = self.params or {}
   self.grad_params = self.grad_params or {}
   return self.params, self.grad_params
end

function NoAttention:zeroGradParameters()

end

