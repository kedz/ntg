local SoftMax = torch.class('ntg.network.SoftMax')

function SoftMax:__init()
    self.op = nn.SoftMax()
end

function SoftMax:forward(input, mask)

   
   assert(input:dim() == 2 or input:dim() == 3)

   if input:dim() == 3 then

      local T = input:size(1)
      local N = input:size(2)
      local V = input:size(3)
      local TN = T * N


      if mask then
         if mask:dim() == 2 then

            self.output = self.op:forward(input:view(TN, V)):view(T, N, V)
            local mask3d = mask:view(T, N, 1):expand(T, N, V)
            self.output:maskedFill(mask3d, 0)

         else

            -- mask dim is 3
            input:maskedFill(mask, -math.huge)
            self.output = self.op:forward(input:view(TN, V)):view(T, N, V)
            -- mask nans from whole rows zeroed out.
            self.output[self.output:ne(self.output)] = 0

         end
      else
         -- no mask
         self.output = self.op:forward(input:view(T * N, V)):view(T, N, V)
      end

   else

      local N = input:size(1)
      local V = input:size(2)

      if mask then
         if mask:dim() == 1 then

            local mask2d = mask:view(N, 1):expand(N, V)
            
            self.output = self.op:forward(input)
            self.output:maskedFill(mask2d, 0)
         else 
             -- mask dim is 2d
            
            input:maskedFill(mask, -math.huge)
            self.output = self.op:forward(input)
            -- mask nans from whole rows zeroed out.
            self.output[self.output:ne(self.output)] = 0
         end
      else
         -- no mask
         self.output = self.op:forward(input)
      end

   end

   return self.output  

end

function SoftMax:backward(input, grad_output, mask)
   
   assert(input:dim() == 2 or input:dim() == 3)

   if input:dim() == 3 then

      local T = input:size(1)
      local N = input:size(2)
      local V = input:size(3)
      local TN = T * N
 

      if mask then
         if mask:dim() == 2 then

            local mask3d = mask:view(T, N, 1):expand(T, N, V)
            self.grad_input = self.op:backward(
               input:view(TN, V), grad_output:view(TN, V)):view(T, N, V)
            self.grad_input:maskedFill(mask3d, 0)

         else
            -- mask dim == 3
             
            grad_output:maskedFill(mask, 0)
            self.grad_input = self.op:backward(
               input:view(TN, V), grad_output:view(TN, V)):view(T, N, V)

         end
      else
         -- no mask
         self.grad_input = self.op:backward(
            input:view(TN, V), grad_output:view(TN, V)):view(T, N, V)
      end

   else

      local N = input:size(1)
      local V = input:size(2)
      
      if mask then
         if mask:dim() == 1 then
            local mask2d = mask:view(N, 1):expand(N, V)
            self.grad_input = self.op:backward(input, grad_output)
            self.grad_input:maskedFill(mask2d, 0)
         else
            -- mask dim == 2
            grad_output:maskedFill(mask, 0)
            self.grad_input = self.op:backward(input, grad_output)

         end
      else
         -- no mask
         self.grad_input = self.op:backward(input, grad_output)
      end
   end

   return self.grad_input

end


