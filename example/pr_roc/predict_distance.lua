
require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
local Threads = require 'threads'
package.path = '../../?.lua;'..package.path
local parallel_utils = require 'utils.parallel_utils'
local fashion_pair_utils= require 'utils.fashion_pair_utils'


local gpus = {1}
local metric = 'L2'
local model_path = 
  '/data2/ImageNet/ILSVRC2012/torch_cache/X_gpu1_resception_nag_lr0.00450_decay_start0_every160000/'
local model_filename = 
  paths.concat(model_path, 'model_29.bn_removed.t7')
  --paths.concat(model_path, 'model_29.t7')
print(string.format( '===> Loading model: %s', model_filename))
local model = torch.load(model_filename)
model:remove(#model.modules)
model:remove(#model.modules)
if metric == 'L2' then
  model:add(nn.Normalize(2))
elseif metric == 'L1' then
  model:add(nn.Normalize(1))
end
model:evaluate()
cudnn.fastest = false
cudnn.benchmark = false
cudnn.verbose = false
if #gpus > 1 then
  model = parallel_utils.makeDataParallel(model, gpus)
else model:cuda() end
print(model)
collectgarbage()

local logger = optim.Logger(string.format(
  '%s.distance.%s.log', model_filename, metric))

print '===> Loading mean, std' 
local mean_std = torch.load(
  '/data2/ImageNet/ILSVRC2012/torch_cache/meanstdCache.t7'
)

local image_list_q, image_list_ref, label_list = fashion_pair_utils.get_test()

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


local batch_timer= torch.Timer()
local data_timer = torch.Timer()
local testBatch = function(n, inputs_q, inputs_ref, label)
  local elapsed_loading = data_timer:time().real
  batch_timer:reset()

  local output_q  = model:forward(inputs_q:cuda()):clone()
  cutorch.synchronize()
  local output_ref= model:forward(inputs_ref:cuda())
  cutorch.synchronize()

  output_q  = output_q:mean(1):squeeze()
  output_ref= output_ref:mean(1):squeeze()

  local distance
  if metric == 'L2' then
    distance = math.sqrt(torch.pow(output_q - output_ref, 2.0):sum())
  elseif metric == 'L1' then
    distance = torch.abs(output_q - output_ref):sum()
  end
  logger:add{
    ['distance'] = distance,
    ['label'] = label
  }
  print(string.format(
    '%d %f %d %f %f', 
    n, distance, label, batch_timer:time().real, elapsed_loading))
  data_timer:reset()

  if n % 500 == 0 then collectgarbage() end
end


local global_timer = torch.Timer()
for n, filename in ipairs(image_list_q) do
  donkeys:addjob(
    function()
      local query_filename = filename
      local ref_filename = image_list_ref[n]
      local label = label_list[n]
      local img_q  = loader.inception7_aug10(
        query_filename, loadSize, sampleSize, mean_std.mean, mean_std.std)
      local img_ref= loader.inception7_aug10(
        ref_filename, loadSize, sampleSize, mean_std.mean, mean_std.std)
      return n, img_q, img_ref, label
    end,
    testBatch
  )
end
donkeys:synchronize()
cutorch.synchronize()

local elapsed_global = global_timer:time().real
print(('elasped: %.4f'):format(elapsed_global))


