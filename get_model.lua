require('nngraph')
require('cunn')
require('cudnn')
require('nnx')


local function getConvNet(struct, nChannels, h, w, nOutputChannels, nOutputElements)
    local isInFCMode, nElements = false, nil
    local input = nn.Identity()()
    local x = nn.Identity()(input)
    local feature = nil
    for i = 1, #struct do
        if struct[i][1] == 'conv' then
            local nOutputs = struct[i][3] or nOutputChannels
            assert(not isInFCMode) -- no convolutions after FC
            assert(nOutputs ~= nil) -- no nil if nOutputChannels is nil
            assert((struct[i][3] ~= nil) or (i == #struct)) -- no nil except in last layer
            x = cudnn.SpatialConvolution(nChannels, nOutputs,
                                         struct[i][2], struct[i][2],
                                         struct[i][4], struct[i][4]):cuda()(x)
            if struct[i][4] ~= nil then
                nChannels, h, w = nOutputs, math.floor((h - struct[i][2])/struct[i][4]) + 1, math.floor((w - struct[i][2])/struct[i][4]) + 1
            else
                nChannels, h, w = nOutputs, h - struct[i][2] + 1, w - struct[i][2] + 1
            end
        elseif struct[i][1] == 'convp' then
            local nOutputs = struct[i][3] or nOutputChannels
            assert(struct[i][2] % 2 == 1) -- no even kernel sizes when padding!
            assert(not isInFCMode) -- no convolutions after FC
            assert(nOutputs ~= nil) -- no nil if nOutputChannels is nil
            assert((struct[i][3] ~= nil) or (i == #struct)) -- no nil except in last layer
            x = cudnn.SpatialConvolution(nChannels, nOutputs,
                                         struct[i][2], struct[i][2],
                                         1, 1, (struct[i][2]-1)/2,
                                         (struct[i][2]-1)/2):cuda()(x)
            nChannels = nOutputs
        elseif struct[i][1] == 'maxpool' then
            assert(not isInFCMode) -- no pooling after FC
            x = cudnn.SpatialMaxPooling(struct[i][2], struct[i][2],
                                        struct[i][3], struct[i][3])(x)
            h = math.floor((h - struct[i][2])/struct[i][3] + 1)
            w = math.floor((w - struct[i][2])/struct[i][3] + 1)
        elseif struct[i][1] == 'fc' then
            local nOutputs = struct[i][2] or nOutputElements
            assert(nOutputs ~= nil) -- no nil if nOutputElements is nil
            assert((struct[i][2] ~= nil) or (i == #struct)) -- no nil except in last layer
            if not isInFCMode then
                nElements = h*w*nChannels
                x = nn.View(nElements):setNumInputDims(3)(x)
                isInFCMode = true
            end
            x = nn.Linear(nElements, nOutputs):cuda()(x)
            nElements = nOutputs
        elseif struct[i][1] == 'feature' then
            assert(feature == nil) -- only one feature layer (for now)
            feature = x
        elseif struct[i][1] == 'spatialbatchnorm' then
            x = nn.SpatialBatchNormalization(nChannels)(x)
        else
            error('Unknown network element ' .. struct[i][1])
        end
        if i ~= #struct then
            x = nn.ReLU()(x)
        end
    end
    local net = nn.gModule({input}, {x, feature})
    if isInFCMode then
        return net, nElements
    else
        return net, nChannels, h, w
    end
end

function getPyrModel(opt, dataset, in_modules)
    -- assume input/target is between -1 and 1
    local out_modules = {}
    local function getPred(imagesScaled, inputGuess, scale, scaleRatio, in_module)
        -- input: images(scale res), guess(scale/2 res)
        local ws, hs = opt.w / scale, opt.h / scale
        local guessScaled, x = nil, nil
        local nInputChannels = opt.nInputFrames*dataset.nChannels
        if inputGuess ~= nil then
            guessScaled = nn.SpatialUpSamplingNearest(scaleRatio)(inputGuess)
            nInputChannels = nInputChannels +opt.nTargetFrames*dataset.nChannels
	          x = nn.JoinTable(2){imagesScaled, guessScaled}
	      else
	          x = imagesScaled
        end
      	local mod = in_module
      	if not mod then
      	   mod = getConvNet(opt.modelStruct[scale], nInputChannels,
      			  hs, ws, opt.nTargetFrames*dataset.nChannels)
      	end
      	mod = mod:cuda()
      	x = mod(x)
      	out_modules[scale] = mod
        local x, features = x:split(2)
        if inputGuess ~= nil then
            x = nn.CAddTable(){x, guessScaled}
        end
        x = nn.Tanh()(x)
        return x, features
    end

    local inputImages = nn.Identity()()
    local pred, features = {}, {}
    for i = 1, #opt.scaleList do
        local scale = opt.scaleList[i]
    	  local mod = nil
    	  if in_modules then
    	     mod = in_modules[scale]
	      end
        pred[i], features[i] =
            getPred(nn.SelectTable(i)(inputImages),
            pred[i-1], --nil if i == 0, on purpose
            scale,
            (i == 1) or (opt.scaleList[i-1] / scale),
		        mod)
    end
    pred = nn.Identity()(pred)
    features = nn.Identity()(features)
    model = nn.gModule({inputImages}, {pred, features})
    model = model:cuda()
    return model, out_modules
end

function getRecModel(opt, model, datasource)
   assert(opt.h == opt.w)
   local input = nn.Identity()()
   local output = {}
   local lastinput = input
   for i = 1, opt.nRec do
      local netoutput = model:clone('weight', 'bias', 'gradWeight', 'gradBias')(lastinput)
      netoutput = nn.SelectTable(1)(netoutput)
      output[i] = netoutput
      if i ~= opt.nRec then
	 local newinput = {}
	 for j = 1, #opt.scaleList do
	    local npix = opt.h / opt.scaleList[j]
	    local x1 = nn.SelectTable(j)(lastinput)
	    x1 = nn.View(opt.batchsize, opt.nInputFrames, datasource.nChannels, npix, npix)(x1)
	    x1 = nn.Narrow(2, 2, opt.nInputFrames-1)(x1)
	    local x2 = nn.SelectTable(j)(netoutput)
	    x2 = nn.View(opt.batchsize, 1, datasource.nChannels, npix, npix)(x2)
	    local y = nn.JoinTable(2){x1, x2}
	    newinput[j] =
	       nn.View(opt.batchsize, opt.nInputFrames*datasource.nChannels, npix, npix)(y)
	 end
	 lastinput = newinput
      end
   end
   if #output == 1 then
      local dummy = nn.ConcatTable()
      dummy:add(nn.Identity())
      output = dummy(output)
      return nn.gModule({input}, {output}):cuda()
   else
      return nn.gModule({input}, output):cuda()
   end
end

function getPyrAdv(opt, dataset)
    local inputImages = nn.Identity()()
    local inputPred = nn.Identity()()
    local adv = {}
    for i = 1, #opt.scaleList do
        assert(opt.advStruct[opt.scaleList[i] ] ~= nil) -- model and adv must have same scales
        local x = nn.JoinTable(2){nn.SelectTable(i)(inputImages),
                                  nn.SelectTable(i)(inputPred)}
        x = getConvNet(opt.advStruct[opt.scaleList[i] ],
                       (opt.nInputFrames+opt.nTargetFrames)*dataset.nChannels,
                       opt.w / opt.scaleList[i], opt.h / opt.scaleList[i], nil, 1)(x)
        adv[i] = nn.Sigmoid()(x)
    end

    advmodel = nn.gModule({inputImages, inputPred}, adv)
    advmodel = advmodel:cuda()
    return advmodel
end

function getRecAdv(opt, advmodel, datasource, in_modules)
   assert((advmodel == nil) ~= (in_modules == nil))
   local input1 = nn.Identity()()
   local input2 = nn.Identity()()
   local output = {}
   local input1b = input1
   out_modules = {}
   for i = 1, opt.nRec do
      local input2b = nn.SelectTable(i)(input2)
      local mod = nil
      if advmodel ~= nil then
	 if opt.advshare == true then
	    mod = advmodel:clone('weight', 'bias', 'gradWeight', 'gradBias')
	 else
	    mod = advmodel:clone()
	    print("====================================================================")
	    print("=================       CLONING ADVMODEL       =====================")
	    print("====================================================================")
	 end
      else
	 if in_modules[i] ~= nil then
	    mod = in_modules[i]
	 else
	    if opt.advshare == true then
	       mod = in_modules[#in_modules]:clone('weight', 'bias', 'gradWeight', 'gradBias')
	       print("====================================================================")
	       print("=================     SHARING LAST ADVMODEL    =====================")
	       print("====================================================================")
	    else
	       mod = in_modules[#in_modules]:clone()
	       print("====================================================================")
	       print("=================     CLONING LAST ADVMODEL    =====================")
	       print("====================================================================")
	    end
	 end
      end
      for i, node in ipairs(mod.backwardnodes) do
	 --TODO: somehow :cuda() fails otherwise
	 node.data.gradOutputBuffer = nil
      end
      out_modules[i] = mod
      output[i] = mod{input1b, input2b}
      if i ~= opt.nRec then
	 local newinput1b = {}
	 for j = 1, #opt.scaleList do
	    local npix = opt.h / opt.scaleList[j]
	    local x1 = nn.SelectTable(j)(input1b)
	    x1 = nn.View(opt.batchsize, opt.nInputFrames, datasource.nChannels, npix, npix)(x1)
	    x1 = nn.Narrow(2, 2, opt.nInputFrames-1)(x1)
	    local x2 = nn.SelectTable(j)(input2b)
	    x2 = nn.View(opt.batchsize, 1, datasource.nChannels, npix, npix)(x2)
	    local y = nn.JoinTable(2){x1, x2}
	    newinput1b[j] =
	       nn.View(opt.batchsize, opt.nInputFrames*datasource.nChannels, npix, npix)(y)
	 end
	 input1b = nn.Identity()(newinput1b)
      end
   end
   if #output == 1 then
      local dummy = nn.ConcatTable()
      dummy:add(nn.Identity())
      output = dummy(output)
      return nn.gModule({input1, input2}, {output}):cuda(), out_modules
   else
      return nn.gModule({input1, input2}, output):cuda(), out_modules
   end
end

function getPyrPreprocessor(opt, dataset)
    local net = nn.ConcatTable()
    for i = 1, #opt.scaleList do
        local net2 = nn.Sequential()
        net:add(net2)
        net2:add(nn.FunctionWrapper(
                     function(self) end,
                     function(self, input)
                         return input:view(input:size(1),
                                               -1, input:size(input:dim()-1),
                                           input:size(input:dim()))
                     end,
                     function(self, input, gradOutput)
                         return gradOutput:viewAs(input)
                     end))
        scale = opt.scaleList[i]
        net2:add(nn.SpatialAveragePooling(scale, scale, scale, scale))
    end
    net:cuda()
    return net
end

-- replicated the criterion into a sort of parallel criterion
-- TODO: is this used?
function getPyrCriterion(opt, simpleCriterion)
    local output = {}
    output.criterion = nn.ParallelCriterion()
    for i = 1, #opt.scaleList do
        output.criterion:add(simpleCriterion:clone())
    end
    output.criterion:cuda()
    output.dsnet = nn.ConcatTable()
    for i = 1, #opt.scaleList do
        local scale = opt.scaleList[i]
        output.dsnet:add(nn.SpatialAveragePooling(scale, scale, scale, scale))
    end
    output.dsnet:cuda()
    function output:forward(input, target)
        return output.criterion:forward(input, output.dsnet:forward(target))
    end
    function output:updateGradInput(input, target)
        return output.criterion:backward(input, output.dsnet.output)
    end
    output.backward = output.updateGradInput
    return output
end

GDL, gdlparent = torch.class('nn.GDLCriterion', 'nn.Criterion')

function GDL:__init(alpha)
    gdlparent:__init(self)
    self.alpha = alpha or 1
    assert(alpha == 1) --for now
    local Y = nn.Identity()()
    local Yhat = nn.Identity()()
    local Yi1 = nn.SpatialZeroPadding(0,0,0,-1)(Y)
    local Yj1 = nn.SpatialZeroPadding(0,0,-1,0)(Y)
    local Yi2 = nn.SpatialZeroPadding(0,-1,0,0)(Y)
    local Yj2 = nn.SpatialZeroPadding(-1,0,0,0)(Y)
    local Yhati1 = nn.SpatialZeroPadding(0,0,0,-1)(Yhat)
    local Yhatj1 = nn.SpatialZeroPadding(0,0,-1,0)(Yhat)
    local Yhati2 = nn.SpatialZeroPadding(0,-1,0,0)(Yhat)
    local Yhatj2 = nn.SpatialZeroPadding(-1,0,0,0)(Yhat)
    local term1 = nn.Abs()(nn.CSubTable(){Yi2, Yi1})
    local term2 = nn.Abs()(nn.CSubTable(){Yhati2,  Yhati1})
    local term3 = nn.Abs()(nn.CSubTable(){Yj2, Yj1})
    local term4 = nn.Abs()(nn.CSubTable(){Yhatj2, Yhatj1})
    local term12 = nn.CSubTable(){term1, term2}
    local term34 = nn.CSubTable(){term3, term4}
    self.net = nn.gModule({Yhat, Y}, {term12, term34})
    self.net:cuda()
    self.crit = nn.ParallelCriterion()
    self.crit:add(nn.AbsCriterion())
    self.crit:add(nn.AbsCriterion())
    self.crit:cuda()
    self.target1 = torch.CudaTensor()
    self.target2 = torch.CudaTensor()
end

function GDL:updateOutput(input, target)
    self.netoutput = self.net:updateOutput{input, target}
    self.target1:resizeAs(self.netoutput[1]):zero()
    self.target2:resizeAs(self.netoutput[2]):zero()
    self.target = {self.target1, self.target2}
    self.loss = self.crit:updateOutput(self.netoutput, self.target)
    return self.loss
end

function GDL:updateGradInput(input, target)
    local gradInput =
        self.crit:updateGradInput(self.netoutput, self.target)
    self.gradInput =
        self.net:updateGradInput({input, target}, gradInput)[1]
    return self.gradInput
end
