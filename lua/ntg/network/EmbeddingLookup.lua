local EmbeddingLookup, parent = torch.class('ntg.network.EmbeddingLookup',
   'ntg.network.Network')

function EmbeddingLookup:__init(vocabSize, embeddingSize)
   self.vocabSize = vocabSize
   self.embeddingSize = embeddingSize
   self.lookup = nn.LookupTable(vocabSize, embeddingSize)
end

function EmbeddingLookup:reset()
   self.lookup:reset()
   return self
end

function EmbeddingLookup:description(offset)
  offset = offset or ''
  local totalParams = self.vocabSize * self.embeddingSize
  local buffer = offset .. 'Network: EmbeddingLookup\n'
     .. offset .. '|      vocab size: ' .. self.vocabSize .. '\n'
     .. offset .. '|  embedding size: ' .. self.embeddingSize .. '\n'
     .. offset .. '|    total params: ' .. totalParams .. '\n'
     .. offset .. '+-------------------------------------+' 
  return buffer
end


function EmbeddingLookup:zeroGradParameters()
   self.lookup:zeroGradParameters()
end
function EmbeddingLookup:parameters() 
   return {self.lookup.weight}, {self.lookup.gradWeight}
end

function EmbeddingLookup:type(typeString)
   self.lookup = self.lookup:type(typeString)
end

function EmbeddingLookup:forward(input, mask)
   -- batch size
   local N = input:size(2)
   -- max timesteps
   local T = input:size(1)

   if mask then
      input:maskedFill(mask, 1)
   end
   
   self.output = self.lookup:forward(input)

   if mask then
      local mask3d = mask:view(T, N, 1):expand(T, N, self.embeddingSize)
      self.output:maskedFill(mask3d, 0)
      input:maskedFill(mask, 0)
   end
   
   return self.output

end

function EmbeddingLookup:backward(input, gradOutput, mask)
   -- batch size
   local N = input:size(2)
   -- max timesteps
   local T = input:size(1)

   if mask then
      -- technically gradOuput should be zero where there are masked 
      -- inputs, better to be safe than sorry. 
      local mask3d = mask:view(T, N, 1):expand(T, N, self.embeddingSize)
      gradOutput:maskedFill(mask3d, 0)
   end

   self.lookup:backward(input, gradOutput)
 
end
