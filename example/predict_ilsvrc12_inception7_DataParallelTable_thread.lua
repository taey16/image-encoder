
require 'torch'
require 'cutorch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
local Threads = require 'threads'
paths.dofile('../utils/util.lua')
paths.dofile('../utils/imagenet_utils.lua')
paths.dofile('../utils/image_utils.lua')

torch.setnumthreads(4)

print '===> Loading model'
local model_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_30.bn_removed.t7'
local original_model = torch.load(model_filename)
local feature_encoder = original_model:get(1)
local classifier = original_model:get(2)
classifier.modules[#classifier.modules] = nil
classifier:add(cudnn.SoftMax())
local model = nn.Sequential()
model:add(feature_encoder):add(classifier)

local replica = model
model = nn.DataParallelTable(1)
for gpu_id=1,2 do
  cutorch.setDevice(gpu_id)
  model:add(replica:clone():cuda(), gpu_id)
end
cutorch.setDevice(1)
  
print(model)
--model:cuda()
model:evaluate()

print '===> Load classes conf.'
local class_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/classes.t7'
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

local loadSize = {3, 292, 292}
local sampleSize={3, 256, 256}

local top1 = 0
local top5 = 0
local trials = 0

local donkeys = Threads( 
  4, 
  function() require 'torch' end,
  function(thread_index)
    local tid = thread_index
    local seed= 123 + tid
    torch.manualSeed(seed)
    print(('===> Starting donkey with id: %d seed: %d'):format(tid, seed))
  end
)

local timer = torch.Timer()

print('Start')
for n, fname in ipairs(image_list) do
  donkeys:addjob(
    function()
      print('In')
      local filename = paths.concat(dataset_root, fname)
      local label = tonumber(label_list[n]) + 1
      print(filename)

      local start_loading = timer:time().real
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
      return n, sendTensor(data)
    end,
    testBatch
  )

  donkeys:synchronize()
  cutorch.synchronize()
end



local inputsCPU = torch.FloatTensor()
local inputs = torch.CudaTensor()


function testBatch(n, inputsThread)
  cutorch.synchronize()
  collectgarbage()

  local scores, classes
  local elapsed_loading = timer:time().real - start_loading
  local start_process = timer:time().real

  receiveTensor(inputsThread, inputsCPU)
  inputs:resize(inputsCPU:size()):copy(inputsCPU)

  scores = model:forward(inputs:cuda()):float()
  --scores = model_bn:forward(input:cuda()):float()
  scores, classes = torch.mean(scores,1):view(-1):sort(true)
  local elapsed_process = timer:time().real - start_process
  --print(class_conf[classes[1]])
  --print(synset_list[n])

  -- Propagate through the modelwork and 
  -- sort outputs in decreasing order and show 5 best classes
  -- _,classes = model:forward(I):view(-1):float():sort(true)

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
    print(("%d top1: %d/%d = %.5f, top5: %d/%d = %.5f %.4f(%.3f)"):format(
      n, top1 , trials, top1 / trials * 100, top5, trials, top5 / trials * 100,
      elapsed_process, elapsed_loading )
    )
  )
end


