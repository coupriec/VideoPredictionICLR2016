require 'datasources.datasource'
require 'paths'
require 'image'
require 'math'

local function round(x)
   return math.floor(x+0.5)
end

local AugmentDatasource, parent = torch.class('AugmentDatasource', 'ClassDatasource')

function AugmentDatasource:__init(datasource, params)
   parent.__init(self)
   self.datasource = datasource
   self.nChannels, self.nClasses = datasource.nChannels, datasource.nClasses
   if params.crop then
      assert(#(params.crop) == 2)
      self.h, self.w = params.crop[1], params.crop[2]
   else
      self.h, self.w = datasource.h, datasource.w
   end

   if self.datasource.tensortype == 'torch.CudaTensor' then
      print("Warning: AugmentDatasource used with a cuda datasource. Might break")
   end

   self.params = {
      flip = params.flip or 0, --1 for vflip, 2 for hflip, 3 for both
      crop = params.crop or {self.h, self.w},
      scaleup = params.scaleup or 1,
      rotate = params.rotate or 0,
      cropMinimumMotion = params.cropMinimumMotion or nil,
      cropMinimumMotionNTries = params.cropMinimumMotionNTries or 25,
   }
end

local function flatten3d(x)
   -- if x is a video, flatten it
   if x:dim() == 4 then
      return x:view(x:size(1)*x:size(2), x:size(3), x:size(4))
   else
      assert(x:dim() == 3)
      return x
   end
end

local function dimxy(x)
   assert((x:dim() == 3) or (x:dim() == 4))
   if x:dim() == 4 then
      return 3, 4
   else
      return 2, 3
   end
end

local flip_out1, flip_out2 = torch.Tensor(), torch.Tensor()
local function flip(patch, mode)
   local out = patch
   if (mode == 1) or (mode == 3) then
      if torch.bernoulli(0.5) == 1 then
	 flip_out1:typeAs(out):resizeAs(out)
	 image.vflip(flatten3d(flip_out1), flatten3d(out))
	 out = flip_out1
      end
   end
   if (mode == 2) or (mode == 3) then
      if torch.bernoulli(0.5) == 1 then
	 flip_out2:typeAs(out):resizeAs(out)
	 image.hflip(flatten3d(flip_out2), flatten3d(out))
	 out = flip_out2
      end
   end
   return out
end

local function crop(patch, hTarget, wTarget, minMotion, minMotionNTries)
   local dimy, dimx = dimxy(patch)
   local h, w = patch:size(dimy), patch:size(dimx)
   assert((h >= hTarget) and (w >= wTarget))
   if (h == hTarget) and (w == wTarget) then
      return patch
   else
      if minMotion then
	 assert(patch:dim() == 4)
	 local x, y
	 for i = 1, minMotionNTries do
	    y = torch.random(1, h-hTarget+1)
	    x = torch.random(1, w-wTarget+1)
	    local cropped = patch:narrow(dimy, y, hTarget):narrow(dimx, x, wTarget)
	    if (cropped[-1] - cropped[-2]):norm() > math.sqrt(minMotion * cropped[-1]:nElement()) then
	       break
	    end
	 end
	 return patch:narrow(dimy, y, hTarget):narrow(dimx, x, wTarget)
      else
	 local y = torch.random(1, h-hTarget+1)
	 local x = torch.random(1, w-wTarget+1)
	 return patch:narrow(dimy, y, hTarget):narrow(dimx, x, wTarget)
      end
   end
end

local scaleup_out = torch.Tensor()
local function scaleup(patch, maxscale, mode)
   mode = mode or 'bilinear'
   local dimy, dimx = dimxy(patch)
   assert(maxscale >= 1)
   local h, w = patch:size(dimy), patch:size(dimx)
   local maxH, maxW = round(h*maxscale), round(w*maxscale)
   if (maxH == h) and (maxW == w) then
      return patch
   else
      local scaleH = torch.random(h, maxH)
      local scaleW = torch.random(w, maxW)
      if patch:dim() == 3 then
	 scaleup_out:typeAs(patch):resize(patch:size(1), scaleH, scaleW)
      else
	 scaleup_out:typeAs(patch):resize(patch:size(1), patch:size(2), scaleH, scaleW)
      end
      return image.scale(flatten3d(scaleup_out), flatten3d(patch), mode)
   end
end

local rotate_out = torch.Tensor()
local function rotate(patch, thetamax, mode)
   mode = mode or 'bilinear'
   assert(thetamax >= 0)
   if thetamax == 0 then
      return patch
   else
      local theta = torch.uniform(-thetamax, thetamax)
      rotate_out:typeAs(patch):resizeAs(patch)
      return image.rotate(flatten3d(rotate_out), flatten3d(patch), theta, mode)
   end
end

local input2_out = torch.Tensor()
function AugmentDatasource:nextBatch(batchSize, set)
   local input, target = self.datasource:nextBatch(batchSize, set)
   if input:dim() == 4 then
      input2_out:resize(batchSize, input:size(2),
			self.params.crop[1], self.params.crop[2])
   else
      input2_out:resize(batchSize, input:size(2), input:size(3),
			self.params.crop[1], self.params.crop[2])      
   end
   for i = 1, batchSize do
      local x = input[i]
      x = flip(x, self.params.flip)
      x = rotate(x, self.params.rotate)
      x = scaleup(x, self.params.scaleup)
      x = crop(x, self.params.crop[1], self.params.crop[2],
	       self.params.cropMinimumMotion, self.params.cropMinimumMotionNTries)
      input2_out[i]:copy(x)
   end
   return self:typeResults(input2_out, target)
end

--This has NO data augmentation (you can't iterate over augmented data, it's infinite)
function AugmentDatasource:orderedIterator(batchSize, set)
   local it = self.datasource:orderedIterator(batchSize, set)
   return function()
      local input, label = it()
      if input ~= nil then
	 return self:typeResults(input, label)
      else
	 return nil
      end
   end
end
