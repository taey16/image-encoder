
require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
cudnn.fastest = true
cudnn.benchmark = true
cudnn.verbose = true
local Threads = require 'threads'
paths.dofile('../utils/util.lua')
local imagenet_utils = paths.dofile('../utils/imagenet_utils.lua')
local parallel_utils = paths.dofile('../utils/parallel_utils.lua')


local model_filename = 
  --'/data2/ImageNet/ILSVRC2012/torch_cache/ILSVRC2012_X_gpu2_cudnn-v5_resception_epoch7_nag_lr0.04500_decay_seed0.940_start0_every80074_iter240220/model_199.t7'
  --'/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_31.bn_removed.t7'
  '/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/model_31.t7'
print(string.format('===> Loading model: %s', model_filename))
--local model = torch.load(model_filename):get(1)
local model = torch.load(model_filename)
cudnn.convert(model, cudnn)
model = parallel_utils.makeDataParallel(model, {1,2})

print(model)
model:cuda()
model:evaluate()
collectgarbage()

print '===> Load classes conf.'
local class_filename = 
  --'/data2/ImageNet/ILSVRC2012/torch_cache/ILSVRC2012_X_gpu2_cudnn-v5_resception_epoch7_nag_lr0.04500_decay_seed0.940_start0_every80074_iter240220/classes.t7'
  '/data2/ImageNet/ILSVRC2012/torch_cache/inception7_residual/digits_gpu1_inception-v3-2015-12-05_lr0.045_Mon_Jan_18_13_23_03_2016/classes.t7'
class_conf = torch.load(class_filename)

print '===> Loading mean, std' 
local mean_std = torch.load(
  --'/storage/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7'
  '/data2/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7'
)

local dataset_root = 
  '/storage/ImageNet/ILSVRC2012/val'
print '===> Loading synsets'
local synset_words = imagenet_utils.load_synset()
local image_list, label_list, synset_list = imagenet_utils.get_val()

local loadSize = {3, 342, 342}
local sampleSize={3, 299, 299}

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
      img = loader.inception_v3_aug20(filename, loadSize, sampleSize, mean_std.mean, mean_std.std)
      return n, img, label
    end,
    testBatch
  )
end

donkeys:synchronize()
cutorch.synchronize()

local elapsed_global = global_timer:time().real
print(('elasped: %.4f'):format(elapsed_global))


