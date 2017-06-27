local Seq2Seq = torch.class('ntg.model.Seq2Seq')

function Seq2Seq:__init(encoder, bridge, decoder)

   self.encoder = encoder
   self.bridge = bridge
   self.decoder = decoder
   self._modules = {encoder, bridge, decoder}

   self:allocate_memory()

end

function Seq2Seq:allocate_memory()
   local params, grad_params = {}, {}
   for i=1,#self._modules do
      local mod_params, mod_grad_params = self._modules[i]:parameters()
      ntg.util.Table.append(params, mod_params)
      ntg.util.Table.append(grad_params, mod_grad_params)
   end
   self.params = nn.Module.flatten(params)
   self.grad_params = nn.Module.flatten(grad_params)
end

function Seq2Seq:parameters()
   return self.params, self.grad_params
end


function Seq2Seq:description(offset)
   offset = offset or ''
   local buffer = offset .. 'Model: Seq2Seq\n'
      .. offset .. self.encoder:description(offset .. "| ") .. '\n'
      .. offset .. self.decoder:description(offset .. "| ")
   return buffer
end


function Seq2Seq:forward(batch)
   self.encoder_state, self.context = self.encoder:forward(batch)
   
   self.bridge_state = self.bridge:forward(self.encoder_state)
   self.output = self.decoder:forward(
      self.bridge_state, self.context, batch)
   return self.output
end

function Seq2Seq:backward(batch, grad_output)
   self.grad_bridge_state, self.grad_context = self.decoder:backward(
      self.bridge_state, self.context, batch, grad_output)
   self.grad_encoder_state = self.bridge:backward(
      self.encoder_state, self.grad_bridge_state)
   self.encoder:backward(batch, self.grad_encoder_state, self.grad_context)
end

function Seq2Seq:train(batch)

--   self.sourceInputMask = self.sourceInputMask or torch.Tensor()
--   self.sourceInputMask = self.sourceInputMask:typeAs(batch.sourceInput)
--   self.sourceInputMask:eq(batch.sourceInput:t(), 0)
--   self.srcInputMaskB = self.srcInputMaskB or torch.Tensor()
--   self.srcInputMaskB = self.srcInputMaskB:byte()
--   self.srcInputMaskB = self.srcInputMaskB:resize(self.sourceInputMask:size())
--   batch.sourceInputMaskT = self.srcInputMaskB:copy(self.sourceInputMask)
--
--   self.targetInputMask = self.targetInputMask or torch.Tensor()
--   self.targetInputMask = self.targetInputMask:typeAs(batch.targetInput)
--   self.targetInputMask:eq(batch.targetInput:t(), 0)
--   self.tgtInputMaskB = self.tgtInputMaskB or torch.Tensor()
--   self.tgtInputMaskB = self.tgtInputMaskB:byte()
--   self.tgtInputMaskB = self.tgtInputMaskB:resize(self.targetInputMask:size())
--   batch.targetInputMaskT = self.tgtInputMaskB:copy(self.targetInputMask)


   self.encoderState, self.context = self.encoder:forward(batch)
   self.bridgeState = self.bridge:forward(self.encoderState)


   return self.bridgeState
--   print(self.bridgeState)
--   os.exit()
--
--   print(self.decoderState)
--
--   os.exit()
--   self.decoder:forward(batch)
--
--   
--
--   return self.output

end

function Seq2Seq:reset()
   for i=1,#self._modules do
      self._modules[i]:reset()
   end
end

function Seq2Seq:zeroGradParameters()
   self.grad_params:zero()
end
