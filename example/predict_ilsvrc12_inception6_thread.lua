
require 'torch'
require 'cutorch'
require 'loadcaffe'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
local threads = require 'threads'
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')
paths.dofile('../utils/util.lua')

torch.setnumthreads(4)
cutorch.setDevice(4)

print '===> Loading model'
local model_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/gpu_2_lr0.045ThuNov2612:23:162015/model_29.t7'
local original_model = torch.load(model_filename)
local feature_encoder = original_model:get(1):get(1)
local classifier = original_model:get(2)
classifier.modules[#classifier.modules] = nil
classifier:add(cudnn.SoftMax())
local model = nn.Sequential()
model:add(feature_encoder):add(classifier)
--model:add(cudnn.SoftMax())
print(model)
model:cuda()
model:evaluate()

print '===> Load classes conf.'
local class_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception6/gpu_2_lr0.045ThuNov2612:23:162015/classes.t7'
class_conf = torch.load(class_filename)
--for i=1, #class_conf do
--  print(class_conf[i])
--end

print '===> Loading mean, std' 
local mean_std = torch.load(
  '/storage/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7')

local dataset_root = 
  '/storage/ImageNet/ILSVRC2012/val'
print '===> Loading synsets'
local synset_words = load_synset()
local image_list, label_list, synset_list = get_val()

local sampleSize = {3, 224, 224}
local loadSize = {3, 256, 256}

local top1 = 0
local top5 = 0
local trials = 0


local nthread = 2
local njob = 4
local msg = "hello from a satellite thread"


local pool = threads.Threads(
  nthread,
  function()
    require 'torch'
    print('0 -- init callback\n')
  end,
  function(threadid)
     print('1 -- start callback threadid ' .. threadid)
     gmsg = msg -- get it the msg upvalue and store it in thread state
  end
)

local inputsCPU = torch.FloatTensor()
local inputs = torch.CudaTensor()

local jobdone = 0
for n, fname in ipairs(image_list) do
  pool:addjob(
    function()
      --print(fname .. ' ' .. label_list[n])
      local filename = paths.concat(dataset_root, fname)
      --label = tonumber(label_list[n]) + 1

      -- Have to resize and convert from RGB to BGR and subtract mean
      local input = loadImage(filename, loadSize, 0)
      local input1= loadImage(filename, loadSize, 1)
      input = mean_std_norm(input, mean_std.mean, mean_std.std)
      input1= mean_std_norm(input1,mean_std.mean, mean_std.std)
      input = augment_image(input, loadSize, sampleSize )
      input1= augment_image(input1,loadSize, sampleSize )
      local data = torch.FloatTensor(20, sampleSize[1], sampleSize[2], sampleSize[3])
      data[{{1 ,10},{},{},{}}] = input
      data[{{11,20},{},{},{}}] = input1
      return sendTensor(data)
    end,
    function(inputsThread)
      receiveTensor(inputsThread, inputsCPU)
      inputs:resize(inputsCPU:size()):copy(inputsCPU)
      scores = model:forward(inputs:cuda()):float()

      trials = trials + 1
      for k=1,5 do
        if k == 1 and class_conf[classes[k]] == synset_list[n] then 
          top1 = top1 + 1
          top5 = top5 + 1
        elseif k > 1  and class_conf[classes[k]] == synset_list[n] then
          top5 = top5 + 1
        end
      end
      io.flush(
        print(("%d top1: %d/%d = %.5f, top5: %d/%d = %.5f"):format(
        n, top1 , trials, top1 / trials * 100, top5, trials, top5 / trials * 100 )
        )
      )
    end
  )
end

pool:synchronize()
