
require 'nn'
require 'cunn'
require 'optim'
paths.dofile('models/init_model_weight.lua')
paths.dofile('utils/parallel_utils.lua')

-- Create Network
local model_filename = opt.netType .. '_' .. opt.backend
local model_filepath = paths.concat('models', model_filename .. '.lua')
assert(paths.filep(model_filepath), 'File not found: ' .. model_filepath)
paths.dofile(model_filepath)
model = {}
feature_encoder = {}
classifier = {}

if opt.retrain then
  assert(paths.filep(opt.retrain), 
    'File not found: ' .. opt.retrain)
  print('===> Loading model from file: ' .. opt.retrain);
  if opt.use_stn then require 'stn' end
  model = torch.load(opt.retrain)
  feature_encoder, classifier = splitDataParallelTable(model)
  --feature_encoder, _ = splitDataParallelTable(model)
  --classifier = nn.Sequential()
  --classifier:add(nn.View(1*1*1024))
  --classifier:add(nn.Linear(1024, 16))
  --classifier:add(cudnn.LogSoftMax())
else
  print('===> Creating model from file: ' .. model_filepath)
  --model = createModel()
  -- ccn2 data-parallel
  feature_encoder, classifier = createModel()
  MSRinit(feature_encoder)
  --MSRinit(model)
end

spanet = {}
if opt.use_stn then
  paths.dofile('models/spatial_transformer.lua')
  spanet = createModel()
  MSRinit(spanet)
  feature_encoder:insert(spanet, 1)
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

