--[[
   Note that it costs time to switch from set (train/test/valid)
   and change the batch size. If you intend to do it a lot, create
   multiple instances of datasources, with constant set/batchSize
   params:
   nDonkeys [4]
--]]

require 'datasources.datasource'
local threads = require 'threads'

local ThreadedDatasource, parent = torch.class('ThreadedDatasource', 'ClassDatasource')

function ThreadedDatasource:__init(getDatasourceFun, params)
   parent.__init(self)
   self.nDonkeys = params.nDonkeys or 4
   --threads.Threads.serialization('threads.sharedserialize') --TODO
   self.donkeys = threads.Threads(self.nDonkeys,
      function(threadid)
	 require 'torch'
	 require 'math'
	 require 'os'
	 torch.manualSeed(threadid*os.clock())
	 math.randomseed(threadid*os.clock()*1.7)
	 torch.setnumthreads(1)
	 threadid_t = threadid
	 datasource_t = getDatasourceFun()
      end)
   self.donkeys:addjob(
      function()
	 return datasource_t.nChannels, datasource_t.nClasses, datasource_t.h, datasource_t.w
      end,
      function(nChannels, nClasses, h, w)
	 self.nChannels, self.nClasses = nChannels, nClasses
	 self.h, self.w = h, w
      end)
   self.donkeys:synchronize()
   self.started = false
   self.output, self.labels = self.output_cpu, self.labels_cpu

   -- TODO? does that overrides the parent __gc?:
   if newproxy then
      --lua <= 5.1
      self.__gc__ = newproxy(true)
      getmetatable(self.__gc__).__gc = 
	 function() self.output = nil end
   else
      self.__gc = function() self.output = nil end
   end
end

function ThreadedDatasource:type(typ)
   parent.type(self, typ)
   if typ == 'torch.CudaTensor' then
      self.output, self.labels = self.output_gpu, self.labels_gpu
   else
      self.output, self.labels = self.output_cpu, self.labels_cpu
   end
end

function ThreadedDatasource:nextBatch(batchSize, set)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(set ~= nil, 'nextBatch: must specify set')
   local function addjob()
      self.donkeys:addjob(
	 function()
	    collectgarbage()
	    local batch, labels = datasource_t:nextBatch(batchSize, set)
	    return batch, labels
	 end,
	 function(outputs, labels)
	    if self.output ~= nil then
	       self.output:resize(outputs:size()):copy(outputs)
	       self.labels:resize(labels:size()):copy(labels)
	       self.last_config = {batchSize, set}
	    end
	 end)
   end
   if not self.started then
      self.donkeys:synchronize()
      self.donkeys:specific(false)
      for i = 1, self.nDonkeys do
	 if self.donkeys:acceptsjob() then
	    addjob()
	 end
      end
      self.started = true
   end

   if self.donkeys:haserror() then
      print("ThreadedDatasource: There is an error in a donkey")
      self.donkeys:terminate()
      os.exit(0)
   end

   self.last_config = {}
   while (self.last_config[1] ~= batchSize) or (self.last_config[2] ~= set) do
      addjob()
      self.donkeys:dojob()
   end
   return self.output, self.labels
end

function ThreadedDatasource:orderedIterator(batchSize, set)
   -- this one doesn't parallelize on more than one thread
   -- (this might be a TODO but seems hard)
   assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
   assert(set ~= nil, 'nextBatch: must specify set')
   self.donkeys:synchronize()
   self.donkeys:specific(true)
   self.started = false
   self.donkeys:addjob(
      1, function()
	 collectgarbage()
	 it_t = datasource_t:orderedIterator(batchSize, set) 
	 end)
   local finished = false
   local function addjob()
      self.donkeys:addjob(
	 1,
	 function()
	    return it_t()
	 end,
	 function(output, labels)
	    if output == nil then
	       finished = true
	    else
	       if self.output ~= nil then --TODO: why is the line useful?
		  self.output:resize(output:size()):copy(output)
		  self.labels:resize(labels:size()):copy(labels)
	       end
	    end
	 end)
   end
   return function()
      self.donkeys:synchronize()
      if finished then
	 self.donkeys:addjob(1, function() it_t = nil collectgarbage() end)
	 self.donkeys:synchronize()
      else
	 addjob()
	 return self.output, self.labels
      end
   end
end