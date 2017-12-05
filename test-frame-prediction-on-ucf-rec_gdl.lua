--[[
July 2016
Authors: Michael Mathieu, Camille Couprie
Script to test 2 trained models to predict future frames in video from 4
previous ones on a subset of the UCF101 test dataset.
--]]

require('torch')
require('nngraph')
require('image')
--require('fbtorch')
require('gfx.js')
require('cunn')
require('cudnn')

paths.dofile('upsample.lua')
paths.dofile('expand.lua')
--dofile('ucf101.lua')

torch.manualSeed(1)
torch.setnumthreads(4)
iscuda = false
assert(loadfile("image_error_measures.lua"))(iscuda)

opt_default = {
   full = false, -- display previous frames and target, otherwise the prediction
   with_pyr = true,
   with_delta = true,
   with_cuda = true,
   network_dir = 'AdvGDL',
   delay_gif = 25,
   totalNbiters=1,
   nChannels= 3,
   margin = 5, --for display
   nOutputFrames = 1,
   nOutputFramesRec = 2,
   interv = 1,
   flow_im_used=true
}

op = op or {}
for k, v in pairs(opt_default) do
   if op[k] == nil then
      op[k] = v
   end
end

local inputH, inputW = 240, 320
local netsize = 64
opt = {batchsize = 1}

-- loading trained network

local flow_pth = 'UCF101frm10p/'
local predloaded
if op.network_dir=='Adv'  then
  predloaded = torch.load('trained_models/new_adv_big_64_smalladv.t7')
elseif op.network_dir=='AdvGDL'  then
  predloaded = torch.load('trained_models/new_adv_big_gdl_64.t7')
end
local opt = predloaded.opt
local model = predloaded.model
opt.nOutputFrames = 1
opt.batchsize = 1

------------------------------------------------------------------------------
-- init multiscale model with dsnet
   local dsnet = nn.ConcatTable()
   dsnet:add(nn.SpatialAveragePooling(8,8,8,8))
   dsnet:add(nn.SpatialAveragePooling(4,4,4,4))
   dsnet:add(nn.SpatialAveragePooling(2,2,2,2))
   dsnet:add(nn.SpatialAveragePooling(1,1,1,1))
   dsnet:cuda()
   local dsnetInput = dsnet
   local dsnetTarget = dsnet:clone()

--------------------------------------------------------------------------------
-- network size adaptation for models fine-tuned on larger patchs
for i = 1, #model.modules do
  if torch.type(model.modules[i]) == 'nn.ExpandDim' then
    local xH = math.floor(math.sqrt(model.modules[i].k) /netsize * inputH + 0.5)
    local xW = math.floor(math.sqrt(model.modules[i].k) /netsize * inputW + 0.5)
        model.modules[i].k = xH*xW
    end
    if torch.type(model.modules[i]) == 'nn.View' then
      if model.modules[i].numInputDims == 2 then
        local s1 = model.modules[i].size[1]
        local s2 = math.floor(model.modules[i].size[2] /netsize * inputH + 0.5)
        local s3 = math.floor(model.modules[i].size[3] /netsize * inputW + 0.5)
        model.modules[i].size = torch.LongStorage{s1, s2, s3}
        model.modules[i].numElements = s1*s2*s3
        --print(model.modules.size)
      end
    end
end

local delta = {torch.CudaTensor(opt.batchsize, 2):zero(),
               torch.CudaTensor(opt.batchsize, 4):zero(),
               torch.CudaTensor(opt.batchsize, 6):zero(),
               torch.CudaTensor(opt.batchsize, 8):zero()}

------------------------------------------------------------------------------

function display_frames(my_array,nbframes)

   local inter = torch.Tensor(op.nChannels,my_array:size(2),op.margin):fill(1)
   local todisp = torch.Tensor(op.nChannels,my_array:size(2),op.margin):fill(1)
   local todisp2 = torch.Tensor(nbframes,op.nChannels,my_array:size(2),
      my_array:size(3))
   for i = 1, nbframes do
      for j = 1, op.nChannels do
         todisp2[i][j]= my_array[(i-1)*3+j]
      end
      todisp = torch.cat(todisp, todisp2[i], 3)
      todisp = torch.cat(todisp, inter, 3)
    end
   gfx.image(todisp)
end

function save_frames(prediction, nbframes, filename)
   for i = 1, opt.nInputFrames do
      prediction[i]:add(1):div(2)

      image.save(filename..'/pred_'..i..'.png',prediction[i])
    end
    local new_img = torch.Tensor(op.nChannels,inputH, inputW):fill(0)
    new_img[1]:fill(1)
    for i = opt.nInputFrames+1, opt.nInputFrames+op.nOutputFramesRec do
      prediction[i]:add(1):div(2)
      new_img[{{},{3,inputH-2},{3,inputW-2}}]=
        prediction[i][{{},{3,inputH-2},{3,inputW-2}}]
      image.save(filename..'/pred_'..i..'.png',new_img)
    end
end

------------------------------------------------------------------------------
-- Main job

local sum_PSNR=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_err_sharp2=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_SSIM=torch.Tensor(op.nOutputFramesRec):fill(0)
local nbimagestosave = op.nOutputFramesRec+opt.nInputFrames
local array_to_save= torch.Tensor(nbimagestosave,op.nChannels,inputH,inputW)
local target_to_save =
  torch.Tensor(op.nOutputFramesRec,op.nChannels,inputH,inputW)

local input, output, target
local batch=1
local nbvideos = 3783
local nbframes, nbpartvid
local nbvid = torch.Tensor(op.nOutputFramesRec):fill(0)

local index =
  torch.range(1,(opt.nInputFrames+op.nOutputFramesRec)*op.interv, op.interv)


for videoidx = 1,nbvideos,10 do
    --local vid, label --= datasets[set]:nextTestImage(videoidx)
    local vid =
      torch.Tensor(opt.nInputFrames+ op.nOutputFramesRec, op.nChannels, 240,320)
    for fr=1,opt.nInputFrames do
      im_name = flow_pth..videoidx..'/pred_'..fr..'.png'
      vid[fr] = (image.load(im_name))
    end
    for fr = 1,op.nOutputFramesRec do
      im_name = flow_pth..videoidx..'/target_'..fr..'.png'
      vid[fr+opt.nInputFrames] = (image.load(im_name))
    end

    vid:mul(2):add(-1)
    nbframes = vid:size(1)
    nbpartvid = torch.abs(nbframes/opt.nInputFrames)

    local filename_out = op.network_dir..'/'..videoidx
    for ii = 1,op.nOutputFramesRec do

        -- extract the first frames
        input = vid[{{1 , opt.nInputFrames}}]
        for f=1,opt.nInputFrames-ii+1 do
          input[f] = vid[index[ii+f-1]]
        end
        for j=1,ii-1 do
          if j> opt.nInputFrames then break end
          input[opt.nInputFrames+1-j] = array_to_save[ii-j+opt.nInputFrames]
        end
        target = torch.Tensor(op.nOutputFrames, op.nChannels, 240,320)
        for f=1,op.nOutputFrames do
          target[f] = vid[index[opt.nInputFrames+ii+f-1]]
        end

        input = input:view(1, op.nChannels*opt.nInputFrames,
          input:size(3), input:size(4))
        target = target:view(1, op.nChannels*opt.nOutputFrames,
          target:size(3), target:size(4))
        if op.with_pyr == true then
          input = dsnetInput:forward(input:cuda())
          target = dsnetTarget:forward(target:cuda())
        end
        if op.with_delta == true then
          output = model:forward({input, delta})[1]
        elseif op.with_pyr == false then
           output = model:forward(input:cuda())
         else
          output = model:forward(input)
        end
        if op.with_pyr == true then
          output = output[4] -- the largest scale output[1][4]
        end
        output = output:double()
        if op.with_pyr == true then
          input = input[4]
          input = input[{{1},{},{},{}}]:float()
          target = target[4]:double()
        end
        output = output[batch]

        -- replace target and input in same space than the output
        target = target[batch]

        if ii==1 then
          array_to_save[{{1,opt.nInputFrames}}]=input
        end
        array_to_save[opt.nInputFrames+ii]=output -- target

        -- extract moving pixels for SNR computations
        if op.flow_im_used then
            local flow_im_name
            local moutput = torch.Tensor(3,240,320):fill(-1)
            local mtarget = torch.Tensor(3,240,320):fill(-1)
            if ii==1 then
              flow_im_name = flow_pth..videoidx..'/pred_4_flow.png'
            else
              flow_im_name = flow_pth..videoidx..'/target_'..(ii-1)..'_flow.png'
            end

            local flow_im = image.load(flow_im_name)
            local s = output[{{1,3}}]:size()

            for j=1, s[2] do
              for k=1, s[3] do
                  if flow_im[1][j][k]< 0.2 or flow_im[2][j][k]< 0.2
                    or flow_im[3][j][k]< 0.2 then -- moving
                     for i=1,s[1] do
                    moutput[i][j][k] = output[i][j][k]
                    mtarget[i][j][k] = target[i][j][k]
                  end
                end
              end
            end

            local psnr = PSNR(moutput, mtarget)
            if psnr < 50 then
                sum_PSNR[ii] = sum_PSNR[ii]+psnr
                sum_SSIM[ii] = sum_SSIM[ii]+SSIM(moutput, mtarget)
                sum_err_sharp2[ii] = sum_err_sharp2[ii] +
                  computel1difference(moutput, mtarget)
                nbvid[ii] = nbvid[ii]+1
            end
        else
          sum_PSNR[ii] = sum_PSNR[ii]+PSNR(output[{{1,3}}], target[{{1,3}}])
          sum_SSIM[ii] = sum_SSIM[ii]+SSIM(output[{{1,3}}], target[{{1,3}}])
          sum_err_sharp2[ii] = sum_err_sharp2[ii] +
            computel1difference(output[{{1,3}}], target[{{1,3}}])
          nbvid[ii] = nbvid[ii]+1
        end
    end --for ii = 1,op.nOutputFramesRec

    print(filename_out)
    os.execute('mkdir -p "' .. filename_out .. '"; ')
    save_frames(array_to_save, nbimagestosave, filename_out)

    for i= 1,op.nOutputFramesRec do
     print('******** video '..videoidx..', '..i..' th frame pred *************')
     print(string.format("score sharp diff: %.2f",sum_err_sharp2[i]/nbvid[i]))
     print(string.format("PSNR: %.2f",sum_PSNR[i]/nbvid[i]))
     print(string.format("SSIM: %.2f",sum_SSIM[i]/nbvid[i]))
    end

    os.execute('convert $(for ((a=1; a<'..nbimagestosave..
      '; a++)); do printf -- "-delay '..op.delay_gif..' '..filename_out..
      '/pred_%s.png " $a; done;) '..filename_out..'result.gif')

end --for videoidx = 1,nbvideos,10

