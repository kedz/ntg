local Network = torch.class('ntg.network.Network')

function Network:forward()
   print("Method " .. torch.type(self) .. ":forward not implemented!")
   os.exit(0)
end

function Network:backward()
   print("Method " .. torch.type(self) .. ":backward not implemented!")
   os.exit(0)
end

function Network:forwardStep()
   print("Method " .. torch.type(self) .. ":forwardStep not implemented!")
   os.exit(0)
end

function Network:backwardStep()
   print("Method " .. torch.type(self) .. ":backwardStep not implemented!")
   os.exit(0)
end

function Network:parameters()
   print("Method " .. torch.type(self) .. ":parameters not implemented!")
   os.exit(0)
end

function Network:reset()
   print("Method " .. torch.type(self) .. ":reset not implemented!")
   os.exit(0)
end

function Network:type()
   print("Method " .. torch.type(self) .. ":type not implemented!")
   os.exit(0)
end
