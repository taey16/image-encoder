
paths.dofile('models/init_model_weight.lua')
paths.dofile('utils/parallel_utils.lua')


model = {}
feature_encoder = {}
classifier = {}
criterion = {}

if opt.retrain then
  assert(paths.filep(opt.retrain), 
    'File not found: ' .. opt.retrain)
  print('===> Loading model from file: '..opt.retrain);
  -- for single-gpu
  model = torch.load(opt.retrain)
  feature_encoder = model:get(1)
  classifier = model:get(2)
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
  feature_encoder, classifier = createModel()
  MSRinit(feature_encoder)
  MSRinit(classifier)
end

if #opt.nGPU > 1 then
  feature_encoder = makeDataParallel(
    feature_encoder, opt.nGPU, opt.GPU)
  classifier:cuda()
else
  feature_encoder:cuda()
  classifier:cuda()
end

model = nn.Sequential():add(feature_encoder):add(classifier)
criterion = nn.ClassNLLCriterion():cuda()

print(model)
print(criterion)
print('===> Loading model complete')

collectgarbage()

