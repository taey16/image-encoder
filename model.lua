
paths.dofile('models/init_model_weight.lua')
local parallel_utils = require 'utils.parallel_utils'

protos = {}
criterion = {}

if opt.retrain then
  assert(paths.filep(opt.retrain), 
    'File not found: ' .. opt.retrain)
  print('===> Loading model from file: '..opt.retrain);
  -- for single-gpu
  protos.encoder = torch.load(opt.retrain)
  print(protos.encoder)
  protos.encoder.modules[#protos.encoder] = nil
  protos.encoder.modules[#protos.encoder] = nil
  --protos.encoder:get(1):add(nn.View(2048))
  protos.classifier = nn.Sequential()
  protos.classifier:add(nn.Linear(2048,256))
  protos.classifier:add(cudnn.BatchNormalization(256,0.001,nil,true))
  protos.classifier:add(cudnn.ReLU(true))
  protos.classifier:add(nn.Linear(256,256))
  protos.classifier:add(cudnn.BatchNormalization(256,0.001,nil,true))
  protos.classifier:add(cudnn.ReLU(true))
  protos.classifier:add(nn.Linear(256,2))
  protos.classifier:add(cudnn.LogSoftMax())
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

if #opt.nGPU > 1 then
  protos.encoder = parallel_utils.makeDataParallel(protos.encoder, opt.nGPU)
else
  cudnn.fastest, cudnn.benchmark = true, true
end

protos.encoder:cuda()
protos.classifier:cuda()
criterion = nn.ClassNLLCriterion():cuda()

print(protos.encoder)
print(protos.classifier)
print(criterion)
print('===> Loading model complete')

collectgarbage()

