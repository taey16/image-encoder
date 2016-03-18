
paths.dofile('models/init_model_weight.lua')
local parallel_utils = require 'utils.parallel_utils'

model = {}
criterion = {}

if opt.retrain then
  assert(paths.filep(opt.retrain), 
    'File not found: ' .. opt.retrain)
  print('===> Loading model from file: '..opt.retrain);
  -- for single-gpu
  model = torch.load(opt.retrain)
  --print(model)
  -- resception
  model:remove(22)
  model:remove(21)
  model:add(nn.Linear(2048,opt.nClasses))
  model:add(cudnn.LogSoftMax())
  --[[
  -- inception-v3-2015-12-06
  model.modules[#model] = nil
  model:get(1):add(nn.View(2048))
  model:get(1):add(nn.Linear(2048,opt.nClasses))
  model:get(1):add(cudnn.LogSoftMax())
  --]]
  --[[
  -- for inception-v3-2015-12-05
  feature_encoder = torch.load(opt.retrain)
  feature_encoder.modules[#feature_encoder] = nil
  feature_encoder.modules[#feature_encoder] = nil
  feature_encoder.modules[#feature_encoder] = nil
  classifier = nn.Sequential()
  classifier:add(nn.View(2048))
  classifier:add(nn.Linear(2048, 1000))
  classifier:add(cudnn.LogSoftMax())
  cudnn.convert(feature_encoder, cudnn)
  cudnn.convert(classifier, cudnn)
  feature_encoder:cuda()
  classifier:cuda()
  --]]
else
  local model_filename = opt.netType..'.lua'
  local model_filepath = paths.concat('models', model_filename)
  assert(paths.filep(model_filepath), 
    'File not found: '..model_filepath)
  paths.dofile(model_filepath)
  print('===> Creating model from file: '..model_filepath)
  model = createModel()
  MSRinit(model)
end

cudnn.fastest, cudnn.benchmark = true, true
if #opt.nGPU > 1 then
  model = parallel_utils.makeDataParallel(model, opt.nGPU)
end

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

print(model)
print(criterion)
print('===> Loading model complete')

collectgarbage()

