
local function append(dst, src)
   for i=1,#src do
      table.insert(dst, src[i])
   end  
end

return {append=append}
