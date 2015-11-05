
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

if opt.retrain then
  assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
  print('===> Loading model from file: ' .. opt.retrain);
  if opt.use_stn then require 'stn' end
  model = torch.load(opt.retrain)
  -- fine tuning
  --model:remove(#model.modules)
  --model:remove(#model.modules)
  --model:insert(nn.Dropout(0.4))
  --model:insert(nn.Linear(1024,16))
  --model:insert(cudnn.LogSoftMax())
  -- remove dropout
  --model:remove(26)
  --model:insert(nn.Dropout(0.0), 26)
else
  print('===> Creating model from file: ' .. model_filepath)
  model = createModel(opt.nGPU)
end

spanet = {}
if opt.use_stn and not opt.retrain then
  paths.dofile('models/stn_ilsvrc.lua')
  spanet = createModel(opt.nGPU)
  model:insert(spanet, 1)
end

if opt.nGPU > 1 then
   model = makeDataParallel(model, opt.nGPU)
end

criterion = nn.ClassNLLCriterion()

print(model)
print(criterion)
print('===> Converting model to CUDA')
model = model:cuda()
criterion:cuda()
print('===> Loading model complete')

collectgarbage()

