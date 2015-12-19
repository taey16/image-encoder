
paths.dofile('models/init_model_weight.lua')
paths.dofile('utils/parallel_utils.lua')

local model_filename = opt.netType..'.lua'
local model_filepath = paths.concat('models', model_filename)
assert(paths.filep(model_filepath), 
  'File not found: '..model_filepath)
paths.dofile(model_filepath)

model = {}
feature_encoder = {}
classifier = {}
criterion = {}

if opt.retrain then
  assert(paths.filep(opt.retrain), 
    'File not found: ' .. opt.retrain)
  print('===> Loading model from file: '..opt.retrain);
  model = torch.load(opt.retrain)
  feature_encoder, classifier = splitDataParallelTable(model)
else
  print('===> Creating model from file: '..model_filepath)
  feature_encoder, classifier = createModel()
  MSRinit(feature_encoder)
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

