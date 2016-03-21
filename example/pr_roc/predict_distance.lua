
require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
local Threads = require 'threads'
package.path = '../../?.lua;'..package.path
local parallel_utils = require 'utils.parallel_utils'
local attribute_utils= require 'utils.attribute_utils'


local gpus = {1,2}
local attribute_id = 
  --'slit_collar'
  'button'
  --'china_collar'
local model_filename = 
  '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/model_29.bn_removed.t7'
print(string.format('===> Loading model: %s', model_filename))
local model = torch.load(model_filename)
model.remove(#model.modules)
model.remove(#model.modules)
model.remove(#model.modules)
cudnn.fastest = true
cudnn.benchmark = true
cudnn.verbose = false
if #gpus > 1 then
  model = parallel_utils.makeDataParallel(model, gpus)
else
  model:cuda()
end
print(model)
collectgarbage()

print '===> Loading mean, std' 
local mean_std = torch.load(
  '/storage/freebee/attribute_button/torch_cache/meanstdCache.t7'
)

local image_list, label_list = attribute_utils.get_test(attribute_id)

local loadSize = {3, 342, 342}
local sampleSize={3, 299, 299}

local nThreads = 4
local donkeys = Threads( 
  nThreads, 
  function() 
    require 'torch' 
    package.path = '../../?.lua;'..package.path
    loader = require 'donkey.test_donkey'
  end,
  function(thread_index)
    local tid = thread_index
    local seed= 123 + tid
    torch.manualSeed(seed)
    print(('===> Starting donkey with id: %d seed: %d'):format(tid, seed))
  end
)

local top1 = 0
local trials = 0

local batch_timer= torch.Timer()
local data_timer = torch.Timer()
local testBatch = function(n, inputs, label)
  local elapsed_loading = data_timer:time().real
  batch_timer:reset()

  local score, scores, classes
  local output = model.encoder:forward(inputs:cuda())
  cutorch.synchronize()
  score = model.classifier:forward(output):float()
  local elapsed_process = batch_timer:time().real
  score = torch.mean(score,1):view(-1)
  scores, classes = score:sort(true)

  trials = trials + 1
  if classes[1] == label+1 then top1 = top1 + 1 end
  io.flush(print(
    ("%d %s %d top1: %d/%d = %.5f, %.4f(%.3f)"):format(
      n, score[2], label, top1 , trials, top1 / trials * 100,
      elapsed_process, elapsed_loading )
    )
  )
  data_timer:reset()
end


local global_timer = torch.Timer()
for n, filename in ipairs(image_list) do
  donkeys:addjob(
    function()
      local label = label_list[n]
      img = loader.inception7_aug20(
        filename, loadSize, sampleSize, mean_std.mean, mean_std.std)
      return n, img, label
    end,
    testBatch
  )
end
donkeys:synchronize()
cutorch.synchronize()

local elapsed_global = global_timer:time().real
print(('elasped: %.4f'):format(elapsed_global))


