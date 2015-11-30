
require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'stn'
paths.dofile('../utils/image_utils.lua')
paths.dofile('../utils/parallel_utils.lua')

torch.setnumthreads(4)

print '===> Loading model'
local model_filename = 
  '/storage/mnist/mnist_image/torch_cache/inception6/gpu_2TueNov2413:45:572015/model_2.t7'
local model = torch.load(model_filename)
stn = model:get(1):get(1):get(1)
--local feature_encoder, classifier = splitDataParallelTable(model)
--local feature_encoder = ori_model:get(1):get(1)
--local classifier = ori_model:get(2)
--local stn = feature_encoder:get(1):cuda()
--model = {}
--collectgarbage()
--model = nn.Sequential():add(feature_encoder):add(classifier)
print(model)
--print(stn)
--model:cuda()
model:evaluate()

print '===> Loading mean, std' 
local mean_std = torch.load('/storage/mnist/mnist_image/torch_cache/meanstdCache.t7')

local sampleSize = {3, 224, 224}
local loadSize = {3, 256, 256}

local top1 = 0
local top5 = 0
local trials = 0
--for k, fname in ipairs(image_list) do
  --print(fname .. ' ' .. label_list[k])
  --filename = paths.concat(dataset_root, fname)
  --label = tonumber(label_list[k]) + 1
  --filename = '/storage/mnist/mnist_image/val/1/mnist_validation_00000001.png'
  filename = '/storage/mnist/mnist_image/val/0/mnist_validation_00003522.png'
  label = 2

  -- Have to resize and convert from RGB to BGR and subtract mean
  input = loadImage(filename, loadSize)
  save_images(input, 1, 'input.png')
  input = mean_std_norm(input, mean_std.mean, mean_std.std)
  input = augment_image(input, loadSize, sampleSize )
  --input = center_crop(input, sampleSize )
  --input = input:view(10,input:size(1),input:size(2),input:size(3)):repeatTensor(2,1,1)
  scores = stn:forward(input:cuda())
  save_images(scores, 10)
  print(stn.output:size())
  --scores, classes = torch.mean(scores[{1,{}}],1):view(-1):sort(true)
  --print(classes[{{1,5}}])
  --print(scores[{{1,5}}])

  --[[
  -- Propagate through the modelwork and sort outputs in decreasing order and show 5 best classes
  _,classes = model:forward(I):view(-1):float():sort(true)
  --]]

  trials = trials + 1
  top1 = top1 + classes[{{1,1}}]:eq(label):sum()
  top5 = top5 + classes[{{1,5}}]:eq(label):sum()
  io.flush(
    print(("%d top1: %d/%d = %.5f, top5: %d/%d = %.5f"):format(
      k, top1 , trials, top1 / trials * 100, top5, trials, top5 / trials * 100 )
    )
  )

-- end

