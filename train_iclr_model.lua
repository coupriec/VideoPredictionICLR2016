--[[
    Trains an L2 + adversarial network (can be only L2 by setting advweight to 0)
    to predict next frame.
    The network uses a multi-resolution pyramid (hardcoded to 4 levels for now)
    Uses latent variable in additive mode at each level
    Supports sgd and adagrad optimization
--]]

require('torch')
require('optim')
require('get_model')
require 'gfx.js'

nngraph.setDebug(false)
gfx.verbose = false
torch.setnumthreads(2)
torch.manualSeed(1)

opt_default = {
    -- general
    devid = 2,                 -- GPU id
    saveName = 'model.t7',  -- save file name
    loadName = '',
    loadOpt=false,
    dataset = 'ucf101',        -- dataset name
    -- training
    nEpoches = 10000,         -- number of "epoches" per training
    nIters = 100,             -- number of minibatches per "epoch"
    batchsize = 8,            -- number of samples per minibatches
    -- model
    h = 32,
    w = 32,                    -- size of the patches
    modelOptim = 'sgd',        -- delta(adadelta), grad(adagrad) or sgd
    modelConfig = {
        learningRate = 0.02,
        --learningRateDecay = 0,
        --weightDecay = 0,
        --momentum = 0
    },
    nInputFrames = 4,          -- number of *input* frames (excluding target)
    nTargetFrames = 1,         -- number of frames to predict

    modelStruct = {
        [8] = {
            {'convp', 3, 16},
            {'convp', 3, 32},
            {'feature'},
            {'convp', 3, 16},
            {'convp', 3, nil}},
        [4] = {
            {'convp', 5, 16},
            {'convp', 3, 32},
            {'feature'},
            {'convp', 3, 16},
            {'convp', 5, nil}},
        [2] = {
            {'convp', 5, 16},
            {'convp', 3, 32},
            {'convp', 3, 64},
            {'feature'},
            {'convp', 3, 32},
            {'convp', 3, 16},
            {'convp', 5, nil}},
        [1] = {
            {'convp', 7, 16},
            {'convp', 5, 32},
            {'convp', 5, 64},
            {'feature'},
            {'convp', 5, 32},
            {'convp', 5, 16},
            {'convp', 7, nil}}},
    -- adv
    advOptim = 'sgd',          -- see modelOptim
    advConfig = {
        learningRate = 0.02,
    },
    l2weight = 1,               -- L2 weight in the loss
    advweight = 0.01,              -- adversarial weight in the loss
    advNIter = 1,               -- number of adversarial training iterations
    advExt = 'full',            -- extend adv training to fake "real" examples [none|cheap|full]
    advStruct = {
        [8] = {
            {'conv', 3, 32},
            {'fc', 256},
            {'fc', 128},
            {'fc', nil}},
        [4] = {
            {'conv', 3, 32},
            {'conv', 3, 32},
            {'conv', 3, 64},
            {'fc', 256},
            {'fc', 128},
            {'fc', nil}},
        [2] = {
            {'conv', 5, 32},
            {'conv', 5, 32},
            {'conv', 5, 64},
            {'fc', 256},
            {'fc', 128},
            {'fc', nil}},
        [1] = {
            {'conv', 7, 32},
            {'conv', 7, 32},
            {'conv', 5, 64},
            {'conv', 5, 128},
            {'maxpool', 2, 2},
            --TODO: shared weights with two last layers
            {'fc', 256},
            {'fc', 128},
            {'fc', nil}},
    },
}
opt = opt or {}
for k, v in pairs(opt_default) do
    if opt[k] == nil then
        opt[k] = v
    end
end
modelState = nil
advState = nil
assert((opt.advweight == 0) ~= (opt.advNIter ~= 0)) -- if not, it's probably a mistake

cutorch.setDevice(opt.devid)

loaded = {}
if opt.loadName ~= '' then
    loaded = torch.load(opt.loadName)
    model = loaded.model
    advmodel = loaded.advmodel
    if loaded.opt.h ~= opt.h then
        advmodel = nil
    end
end
if opt.loadOpt == true then
   local oldopt = opt
   opt = loaded.opt
   --opt.devid = oldopt.devid
   --opt.saveName = oldopt.saveName
   for k, v in pairs(opt_override) do
      opt[k] = v
   end
end

local w, h = opt.h, opt.w
local winput = w
local hinput = h
if opt.dataset == 'sports1m' then
   error("no sports1m dataset")
elseif opt.dataset == 'ucf101' then
    require('datasources.thread')
    local optt = opt -- need local var, opt is global
    dataset = ThreadedDatasource(
       function()
	  require('datasources.augment')
	  require('datasources.ucf101')
	  local ucfdataset = UCF101Datasource{
	     nInputFrames = optt.nInputFrames+optt.nTargetFrames
	  }
	  return AugmentDatasource(ucfdataset, {crop = {h, w}})
       end, {nDonkeys = 8})
    dataset:cuda()
else
   error("Unknown dataset " .. opt.dataset)
end

opt.scaleList = {}
for k, v in pairs(opt.modelStruct) do
    opt.scaleList[1+#opt.scaleList] = k
end
table.sort(opt.scaleList, function(a,b) return b<a end)

function printOpt(opt)
    for k, v in pairs(opt) do
        if k == 'modelStruct' or k == 'advStruct' then
            print(k .. ':')
            for j = 1, #opt.scaleList do
                local scale = opt.scaleList[j]
                print('', '' .. scale .. ': ===============')
                for i = 1, #v[scale] do
                    print('', '', unpack(v[scale][i]))
                end
            end
        else
            print(k, ':', v)
        end
    end
end
printOpt(opt)

--------------------------------------------------------------------------------
--   Options are processed.
--   Creating the networks and losses (if necessary)
--------------------------------------------------------------------------------

if model == nil then
    if loaded.gen ~= nil then
       print("TODO: cleanup ---------------------------------")
       model = getPyrModel(opt, dataset, loaded.gen)
    else
       model = getPyrModel(opt, dataset)
    end
else
    print("====================================================================")
    print("================= RESTARTING FROM CURRENT MODEL ====================")
    print("====================================================================")
end

if advmodel == nil then
    advmodel = getPyrAdv(opt, dataset)
else
    print("====================================================================")
    print("============== RESTARTING FROM CURRENT ADVERSARIAL =================")
    print("====================================================================")
end

preprocessInput = getPyrPreprocessor(opt, dataset)
preprocessTarget = getPyrPreprocessor(opt, dataset)

l2criterion = nn.ParallelCriterion()
advcriterion = nn.ParallelCriterion()
for i = 1, #opt.scaleList do
    l2criterion:add(nn.MSECriterion())
    advcriterion:add(nn.BCECriterion())
end
l2criterion:cuda()
advcriterion:cuda()

--------------------------------------------------------------------------------
--   Everything is created and initialized.
--   Training/timing routines
--------------------------------------------------------------------------------

enable_timers = true
timers = {
    model_fprop = torch.Timer(),
    model_bprop = torch.Timer(),
    adv_fprop = torch.Timer(),
    adv_bprop = torch.Timer(),
    data = torch.Timer(),
    total = torch.Timer()
}
function reset_timers()
    for k, v in pairs(timers) do
        v:reset()
        v:stop()
    end
end
function print_timers()
    for k, v in pairs(timers) do
        print(k, v:time().real)
    end
end

function time_run(f, timer, ...)
    local output
    if enable_timers then
        cutorch.synchronize()
        timer:resume()
        output = f(...)
        cutorch.synchronize()
        timer:stop()
    else
        output = f(...)
    end
    return output
end
function model_forward(...)
    return time_run(model.forward, timers.model_fprop, model, ...)
end
function model_updateGradInput(...)
    return time_run(model.updateGradInput, timers.model_bprop, model, ...)
end
function model_accGradParameters(...)
    return time_run(model.accGradParameters, timers.model_bprop, model, ...)
end
function adv_forward(...)
    return time_run(advmodel.forward, timers.adv_fprop, advmodel, ...)
end
function adv_updateGradInput(...)
    return time_run(advmodel.updateGradInput, timers.adv_bprop, advmodel, ...)
end
function adv_accGradParameters(...)
    return time_run(advmodel.accGradParameters, timers.adv_bprop, advmodel, ...)
end

local input_g, target_g = torch.CudaTensor(), torch.CudaTensor()
function getBatch(set)
    local batch, label =
    unpack(time_run(function() return {dataset:nextBatch(opt.batchsize, set)} end,
		    timers.data))
    input_g:resize(opt.batchsize, opt.nInputFrames, dataset.nChannels,
		   batch:size(4), batch:size(5))
    input_g:copy(batch:narrow(2, 1, opt.nInputFrames))
    target_g:resize(opt.batchsize, opt.nTargetFrames, dataset.nChannels,
		   batch:size(4), batch:size(5))
    target_g:copy(batch:narrow(2, opt.nInputFrames+1, opt.nTargetFrames))
    return preprocessInput:forward(input_g), preprocessTarget:forward(target_g)
end

function fevaladv(input1, input2, targetAdv)
    local advout = adv_forward({input1, input2}, targetAdv)
    local adverr = advcriterion:forward(advout, targetAdv)
    return adverr
end

function fevalfull(input, targetL2, targetAdv)
    local output = model_forward(input)
    local pred, features = output[1], output[2]
    local l2err = l2criterion:forward(pred, targetL2)
    local adverr = 0
    if targetAdv ~= nil then
        adverr = fevaladv(input, pred, targetAdv)
    end
    return l2err, adverr
end

function bevaladv(input, targetAdv, scaleAdv)
    local dadverr_dadvo =
        advcriterion:updateGradInput(advmodel.output, targetAdv)
    local dadverr_dpred = adv_updateGradInput({input, model.output[1]}, dadverr_dadvo)
    if scaleAdv ~= 0 then
        adv_accGradParameters({input, model.output[1]}, dadverr_dadvo, scaleAdv)
    end
    return dadverr_dpred
end

derr_dfeatures = {}
for i = 1, #opt.scaleList do
    derr_dfeatures[i] = torch.CudaTensor()
end
function bevalfull(input, targetL2, targetAdv, scalePred, scaleAdv)
    local dl2err_dpred = l2criterion:backward(model.output[1], targetL2)
    for i = 1, #dl2err_dpred do
        dl2err_dpred[i]:mul(opt.l2weight)
    end
    derr_dpred = dl2err_dpred
    if targetAdv ~= nil then
        local dadverr_dpred = bevaladv(input, targetAdv, scaleAdv)[2]
        for i = 1, #dadverr_dpred do
            dadverr_dpred[i]:mul(opt.advweight)
            derr_dpred[i] = dl2err_dpred[i]:add(dadverr_dpred[i])
        end
    end
    local features = model.output[2]
    for i = 1, #opt.scaleList do
        derr_dfeatures[i]:resizeAs(features[i]):zero()
    end
    local derr_di = model_updateGradInput(input, {derr_dpred, derr_dfeatures})
    if scalePred ~= 0 then
        model_accGradParameters(input, {derr_dpred, derr_dfeatures}, scalePred)
    end
    return derr_di
end

local modelW, modelDW = model:getParameters()
local advW, advDW = advmodel:getParameters()

advZeros0 = torch.CudaTensor(opt.batchsize):fill(0)
advZeros = {advZeros0, advZeros0, advZeros0, advZeros0}
advOnes0 = torch.CudaTensor(opt.batchsize):fill(1)
advOnes = {advOnes0, advOnes0, advOnes0, advOnes0}

function trainPred()
    local input, target = getBatch('train')
    local feval = function(x)
        assert(x == modelW)
        model:zeroGradParameters()
        local l2err, adverr = fevalfull(input, target, advOnes)
        bevalfull(input, target, advOnes, 1, 0)
        return opt.l2weight*l2err + opt.advweight*adverr, modelDW
    end
    if opt.modelOptim == 'delta' then
        optim.adadelta(feval, modelW, opt.modelConfig, modelState)
    elseif opt.modelOptim == 'grad' then
        optim.adagrad(feval, modelW, opt.modelConfig, modelState)
    else
        optim.sgd(feval, modelW, opt.modelConfig, modelState)
    end
end

local input_adv_tmp = {}
function trainAdv()
    local input, target = getBatch('train')
    local err = 0
    local feval = function(x)
        assert(x == advW)
        advmodel:zeroGradParameters()
        local _, err1 = fevalfull(input, target, advZeros)
        bevaladv(input, advZeros, 1)
        local err2 = fevaladv(input, target, advOnes)
        bevaladv(input, advOnes, 1)
        err = (err1+err2)/2
        if opt.advExt == 'full' then
            local input, target = getBatch('train')
            local err1 = fevaladv(input, target, advOnes)
            bevaladv(input, advOnes, 1)
            for i = 1, #input do
	       input_adv_tmp[i] = input_adv_tmp[i] or torch.CudaTensor()
                input_adv_tmp[i]:resizeAs(input[i]):copy(input[i])
            end
            local input, target = getBatch('train')
            local err2 = fevaladv(input_adv_tmp, target, advZeros)
            bevaladv(input, advZeros, 1)
            err = err/2 + (err1+err2)/4
        elseif opt.advExt == 'cheap' then
            assert(false)--TODO
            --[[TODO
            for i = 1, #input do
                input_adv_tmp[i]:resizeAs(input[i]):copy(input[i])
            end
            for i = 1, #input-1 do
                input[i]:copy(input_adv_tmp[i+1])
            end
            input[input:size(1)]:copy(input_adv_tmp[1])
            local err1 = feval(input, target
            --]]
        end
        return err, advDW
    end
    if opt.advOptim == 'delta' then
        optim.adadelta(feval, advW, opt.advConfig, advState)
    elseif opt.advOptim == 'grad' then
        optim.adagrad(feval, advW, opt.advConfig, advState)
    else
        optim.sgd(feval, advW, opt.advConfig, advState)
    end
    return err
end

for iEpoch = 1, opt.nEpoches do
    reset_timers()
    cutorch.synchronize()
    timers.total:resume()
    local sumAdvErr = 0
    for iIter = 1, opt.nIters do
       xlua.progress(iIter, opt.nIters)
        if opt.advweight ~= 0 then
            for iAdv = 1, opt.advNIter do
                sumAdvErr = sumAdvErr + trainAdv() / opt.advNIter
            end
        end
        trainPred()
    end
    local avgAdvErr = sumAdvErr / (opt.nIters * opt.batchsize)
    print("adv training: " .. avgAdvErr)

    cutorch.synchronize()
    print_timers()

    if iEpoch % 10 == 0 then
        printOpt(opt)
        collectgarbage()
    end
    if iEpoch % 100 == 1 then
        gfx.clear()
    end


    if paths.filep(opt.saveName .. '.backup') then
       os.remove(opt.saveName .. '.backup')
    end
    os.rename(opt.saveName, opt.saveName .. '.backup')
    torch.save(opt.saveName, {model=model, advmodel=advmodel, opt=opt,
                              modelState=modelState, advState=advState})

    -- display
    local input, target = getBatch('test')
    local pred = model:forward(input)[1]
    local predFull = pred[#pred]:clone()
    predFull = predFull:view(predFull:size(1),
                             opt.nTargetFrames, dataset.nChannels,
                             predFull:size(3), predFull:size(4))
    inputFull = input[#input]
    inputFull = inputFull:view(inputFull:size(1),
                               opt.nInputFrames, dataset.nChannels,
                               inputFull:size(3),
                               inputFull:size(4))
    targetFull = target[#target]
    targetFull = targetFull:view(targetFull:size(1),
                                 opt.nTargetFrames, dataset.nChannels,
                                 targetFull:size(3),
                                 targetFull:size(4))
    local todisp = {}
    for i = 1, opt.nInputFrames do
        for j = 1, opt.batchsize do
            todisp[1+#todisp] = inputFull[j][i]
        end
    end
    for i = 1, opt.nTargetFrames do
        --[[for j = 1, opt.batchsize do
            todisp[1+#todisp] = targetFull[j][i]
            end--]]
        for j = 1, opt.batchsize do
            todisp[1+#todisp] = predFull[j][i]
        end
    end

    win = gfx.image(todisp, {win=win, zoom = 1, nperrow = opt.batchsize,
                             legend=opt.saveName, padding=2})
end
