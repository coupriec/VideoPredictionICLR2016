local ClassDatasource = torch.class('ClassDatasource')

function ClassDatasource:__init()
   self.tensortype = torch.getdefaulttensortype()   
   self.output_cpu, self.labels_cpu = torch.Tensor(), torch.LongTensor()
end

function ClassDatasource:center(trainset, sets)
   -- unused, TODO move
   error("shouldn't be used for now")
   if trainset:dim() == 3 then
      local mean = trainset:mean()
      local std = trainset:std()
      for _, set in pairs(sets) do
	 set:add(-mean):div(std)
      end
   else
      assert(trainset:dim() == 4)
      for iChannel = 1, trainset:size(2) do
	 local mean = trainset[{{},iChannel}]:mean()
	 local std = trainset[{{},iChannel}]:std()
	 for _, set in pairs(sets) do
	    set[{{},iChannel}]:add(-mean):div(std)
	 end
      end
   end
end

function ClassDatasource:normalize(trainset, sets, fullset)
   local function getminmax(set)
      if fullset or set:size(1) < 100 then
	 return set:min(), set:max()
      else
	 set = set[{{1,100}}]--:contiguous()
	 return set:min(), set:max()
      end
   end
   -- scales the data between -1 and 1
   if trainset:dim() == 3 then
      -- grayscale
      local mini, maxi = getminmax(trainset)
      for _, set in pairs(sets) do
	 set:add(-mini):mul(2/(maxi-mini)):add(-1)
      end
   else
      -- rgb (or multichannel)
      assert(trainset:dim() == 4)
      for iChannel = 1, trainset:size(2) do
	 local mini, maxi = getminmax(trainset[{{},iChannel}])
	 for _, set in pairs(sets) do
	    set[{{},iChannel}]:add(-mini):mul(2/(maxi-mini)):add(-1)
	 end
      end
   end
end

function ClassDatasource:typeResults(output, labels)
   if self.tensortype == 'torch.CudaTensor' then
      self.output_gpu:resize(output:size()):copy(output)
      self.labels_gpu:resize(labels:size()):copy(labels)
      return self.output_gpu, self.labels_gpu
   else
      return output, labels
   end
end

function ClassDatasource:type(typ)
   self.tensortype = typ
   if typ == 'torch.CudaTensor' then
      self.output_gpu = torch.CudaTensor()
      self.labels_gpu = torch.CudaTensor()
   else
      self.output_cpu = self.output_cpu:type(typ)
      self.output_gpu = nil
      self.labels_gpu = nil
      collectgarbage()
   end
end

function ClassDatasource:cuda()
   self:type('torch.CudaTensor')
end

function ClassDatasource:float()
   self:type('torch.FloatTensor')
end

function ClassDatasource:double()
   self:type('torch.DoubleTensor')
end