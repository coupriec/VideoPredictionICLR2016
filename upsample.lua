require 'nn'
require 'math'

local SpatialUpSample, parent = torch.class('nn.SpatialUpSample', 'nn.Module')

-- for now, assume square input
function SpatialUpSample:__init(inputH, outputH)
   parent.__init(self)
   self.h = inputH
   self.H = outputH
   self.M = torch.zeros(self.H, self.h)
   local s = self.H / self.h
   for k = 1, self.h do
      for x = 1, self.H do
	 local v = math.max(0, 1 - math.abs((x-1) / s - (s-1)/(2*s) - k + 1))
	 self.M[x][k] = v
      end
   end
   -- fix the first and last lines:
   self.M:cdiv(self.M:sum(2):expandAs(self.M))
   self.output = torch.Tensor()
   self.tmp = torch.Tensor()
   self.tmp2 = torch.Tensor()
   self.gradInput = torch.Tensor()
end

function SpatialUpSample:updateOutput(input)
   assert(input:dim() == 4)
   local bsz, nfeature = input:size(1), input:size(2)
   local h, w = input:size(3), input:size(4)
   assert(h == self.h)
   assert(w == self.h)
   self.tmp:resize(bsz*nfeature*h, self.H)
   self.tmp:mm(input:view(bsz*nfeature*h, w), self.M:t())
   self.tmp = self.tmp:view(bsz*nfeature, h, self.H)
   self.tmp2:resize(bsz*nfeature*self.H, h)
   self.tmp2:copy(self.tmp:transpose(2, 3))
   self.tmp:resize(bsz*nfeature*self.H, self.H)
   self.tmp:mm(self.tmp2, self.M:t())
   self.output:resize(bsz, nfeature, self.H, self.H)
   self.output:copy(self.tmp:view(-1, self.H, self.H):transpose(2, 3))
   return self.output
end

function SpatialUpSample:updateGradInput(input, gradOutput)
   local bsz, nfeature = input:size(1), input:size(2)
   local h, w = input:size(3), input:size(4)
   self.tmp:resize(bsz*nfeature*self.H, self.H)
   self.tmp:copy(gradOutput:view(-1, self.H, self.H):transpose(2, 3))
   self.tmp2:resize(bsz*nfeature*self.H, h)
   self.tmp2:mm(self.tmp, self.M)
   self.tmp2 = self.tmp2:view(bsz*nfeature, self.H, h)
   self.tmp:resize(bsz*nfeature*h, self.H)
   self.tmp:copy(self.tmp2:transpose(2,3))
   self.gradInput:resize(bsz*nfeature*h, w)
   self.gradInput:mm(self.tmp, self.M)
   self.gradInput = self.gradInput:view(bsz, nfeature, h, w)
   return self.gradInput
end
