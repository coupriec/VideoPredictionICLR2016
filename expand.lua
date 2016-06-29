require 'nn'

local ExpandDim, parent = torch.class('nn.ExpandDim', 'nn.Module')

-- expand dim d (must be 1 in the input) k times
function ExpandDim:__init(d, k)
   parent:__init(self)
   self.d = d
   self.k = k
   --self.output = torch.Tensor()
   self.gradInput = torch.Tensor()
end
      
function ExpandDim:updateOutput(input)
   assert(input:size(self.d) == 1)
   local dims = input:size():totable()
   dims[self.d] = self.k
   --self.output:resize(unpack(dims))
   --self.output:copy(input:expand(unpack(dims)))
   self.output = input:expand(unpack(dims))
   return self.output
end

function ExpandDim:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:sum(gradOutput, self.d)
   return self.gradInput
end
