
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

--torch.setnumthreads(4)
--cutorch.setDevice(1)

print '===> Loading model'
local model_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_35.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7_residual/gpu2_residual_feature_lr0.045_epoch19_Wed_Dec_30_20_40_18_2015/model_31.bn_removed.t7'
  --'/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/model_30.bn_removed.t7'
local original_model = torch.load(model_filename)
local feature_encoder = original_model:get(1)
local classifier = original_model:get(2)
classifier.modules[#classifier.modules] = nil
classifier:add(cudnn.SoftMax())
local model = nn.Sequential()
model:add(feature_encoder):add(classifier)
--[[
local replica = model
model = nn.DataParallelTable(1)
for gpu_id=1,2 do
  cutorch.setDevice(gpu_id)
  model:add(replica:clone():cuda(), gpu_id)
end
cutorch.setDevice(1)
--]]
  
print(model)
model:cuda()
model:evaluate()
collectgarbage()

print '===> Load classes conf.'
local class_filename = 
  '/storage/ImageNet/ILSVRC2012/torch_cache/inception7/digits_gpu_2_lr0.045SatDec514:08:122015/classes.t7'
class_conf = torch.load(class_filename)

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


local nThreads = 4
local donkeys = Threads( 
  nThreads, 
  function() 
    require 'torch' 
  end,
  function(thread_index)
    local tid = thread_index
    local seed= 123 + tid
    torch.manualSeed(seed)
    print(('===> Starting donkey with id: %d seed: %d'):format(tid, seed))
    loader = paths.dofile('../donkey/test_donkey.lua')
  end
)

local top1 = 0
local top5 = 0
local trials = 0

local batch_timer= torch.Timer()
local data_timer = torch.Timer()
local testBatch = function(n, inputs, labels)

  local elapsed_loading = data_timer:time().real
  batch_timer:reset()

  local scores, classes
  scores = model:forward(inputs:cuda()):float()
  cutorch.synchronize()
  scores, classes = torch.mean(scores,1):view(-1):sort(true)

  local elapsed_process = batch_timer:time().real

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
  data_timer:reset()
end


local global_timer = torch.Timer()
for n, fname in ipairs(image_list) do
  donkeys:addjob(
    function()
      local filename = paths.concat(dataset_root, fname)
      local label = tonumber(label_list[n]) + 1
      --print(filename)
      img = loader.inception7_aug20(filename, loadSize, sampleSize, mean_std.mean, mean_std.std)
      return n, img, label
    end,
    testBatch
  )
end

donkeys:synchronize()
cutorch.synchronize()

local elapsed_global = global_timer:time().real
print(('elasped: %.4f'):format(elapsed_global))


